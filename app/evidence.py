"""Deterministic Evidence Builder.

Raw BM streams -> compact, auditable evidence packages.
NO LLM here: every number must be reproducible from the snapshot.
"""
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from .taxonomy import (NAME_ALIASES, SECTOR_MAP, TOPIC_KEYWORDS, ZH_RE, kw_hit,
                       parse_ts, ticker_info)
from .account import adjusted_equity_series, daily_returns, max_drawdown, total_return
from .action_forecast import build_forecast, digest as forecast_digest
from .attention_capital import attention_capital_digest, build_attention_capital
from .behavior import build_behavior_intent
from .portfolio_metrics import build_portfolio_metrics
from .position_events import build_position_dynamics, position_dynamics_digest
from .trade_metrics import build_trade_metrics

# Backwards-compatible private aliases (this module used to own these tables).
_ticker_info = ticker_info
_Z = ZH_RE

# Version of the equity-curve evidence. Bump when a DEFINITION changes (field
# removed, basis changed) so dimensions that consumed it are recomputed even if
# the resulting numbers happen to be identical. Without this, a definition-only
# change would be invisible to the fingerprint and the old value would be carried
# forward indefinitely. v2: duplicate exposure ratios removed, unrealised P/L
# moved to the last_price-vs-cost_price basis.
ASSET_EVIDENCE_VERSION = "asset-evidence-v2"
# v2: unrealised P/L sign now uses last_price vs cost_price (profit_loss_total is
# 37.9% off in magnitude and must not be aggregated).
POSITION_EVIDENCE_VERSION = "position-evidence-v2"

_ACTION_RE = re.compile(r"View (stock|etf|option|index)\s+([A-Z0-9.\-]+)", re.I)
_TIMEFRAME_RE = re.compile(r"Switch to (\S+)\s+timeframe", re.I)


def _ts(s):
    """Kept as a thin alias: timestamp parsing is shared via taxonomy.parse_ts
    (it must handle offsets, date-only and the compact collector form)."""
    return parse_ts(s)


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def behavior_evidence(snap):
    """BM433 -> viewing/timeframe/research behavior summary."""
    ev = snap["behaviors_6m"]
    if not ev:
        return {"available": False}
    views, timeframes, other = Counter(), Counter(), Counter()
    first_last = [x.get("event_date") for x in ev if x.get("event_date")]
    for x in ev:
        a = x.get("action_text") or ""
        m = _ACTION_RE.search(a)
        if m:
            views[(m.group(2).upper(), m.group(1).lower())] += 1
            continue
        m = _TIMEFRAME_RE.search(a)
        if m:
            timeframes[m.group(1)] += 1
            continue
        other[a] += 1
    tick = {t: c for (t, _), c in views.items()}
    tf_total = sum(timeframes.values())
    short_tf = sum(c for k, c in timeframes.items() if re.match(r"^(day|1d|5m|15m|30m|1h|60m|5min|15min|30min)$", k, re.I))
    return {
        "available": True,
        "event_count": len(ev),
        "active_days": len(set(first_last)),
        "date_range": [min(first_last), max(first_last)] if first_last else None,
        "top_viewed_tickers": [{"ticker": t, "count": c, **_ticker_info(t)}
                               for t, c in sorted(tick.items(), key=lambda kv: -kv[1])[:12]],
        "tickers_viewed": len(tick),
        "etf_views": sum(c for (t, k), c in views.items() if k == "etf"),
        "stock_views": sum(c for (t, k), c in views.items() if k == "stock"),
        "option_views": sum(c for (t, k), c in views.items() if k == "option"),
        "timeframe_switch_count": tf_total,
        "short_timeframe_share": round(short_tf / tf_total, 3) if tf_total else None,
        "top_other_actions": [{"action": a, "count": c} for a, c in other.most_common(8)],
    }


def trade_evidence(snap):
    """BM452 -> executed-trade statistics."""
    tr = [t for t in snap["trades_6m"] if (t.get("execution_state") or "executed") == "executed"]
    if not tr:
        return {"available": False}
    buys = [t for t in tr if t.get("side") == "buy"]
    sells = [t for t in tr if t.get("side") == "sell"]
    b_amt = sum(abs(float(t.get("executed_amount") or 0)) for t in buys)
    s_amt = sum(abs(float(t.get("executed_amount") or 0)) for t in sells)
    tickers = Counter((t.get("ticker") or "").upper() for t in tr)
    lev = sum(c for tk, c in tickers.items() if _ticker_info(tk)["sector"] == "Leveraged ETF")
    by_day = Counter(t.get("trade_day") or t.get("trade_date") for t in tr)
    dates = sorted(d for d in by_day if d)
    gaps = [((_ts(b) - _ts(a)).days) for a, b in zip(dates, dates[1:]) if _ts(a) and _ts(b)]
    return {
        "available": True,
        "trade_count": len(tr),
        "buy_count": len(buys), "sell_count": len(sells),
        "buy_amount": round(b_amt, 2), "sell_amount": round(s_amt, 2),
        "buy_sell_amount_ratio": round(b_amt / s_amt, 2) if s_amt else None,
        "unique_tickers": len(tickers),
        "top_tickers": [{"ticker": tk, "count": c, **_ticker_info(tk)}
                        for tk, c in tickers.most_common(10)],
        "leveraged_etf_trade_share": round(lev / len(tr), 3),
        "trading_days": len(by_day),
        "max_trades_per_day": max(by_day.values()) if by_day else 0,
        "median_gap_between_trading_days": sorted(gaps)[len(gaps) // 2] if gaps else None,
        "date_range": [dates[0], dates[-1]] if dates else None,
    }


def behavior_trade_matching(snap, window_hours=48):
    """Did the user research a ticker shortly before trading it?"""
    tr = [t for t in snap["trades_6m"] if (t.get("execution_state") or "executed") == "executed"]
    ev = snap["behaviors_6m"]
    if not tr or not ev:
        return {"available": False}
    views = defaultdict(list)
    for x in ev:
        m = _ACTION_RE.search(x.get("action_text") or "")
        ts = _ts(x.get("timestamp_et"))
        if m and ts:
            views[m.group(2).upper()].append(ts)
    matched = 0
    for t in tr:
        # real execution time; trade_date is the extraction date, not the trade date
        ts = _ts(t.get("executed_at"))
        if ts is None and t.get("trade_date"):
            ts = _ts(f"{t.get('trade_date')}T12:00")
        tk = (t.get("ticker") or "").upper()
        if not ts or tk not in views:
            continue
        if any(0 <= (ts - v).total_seconds() <= window_hours * 3600 for v in views[tk]):
            matched += 1
    return {
        "available": True,
        "window_hours": window_hours,
        "trades_with_prior_view": matched,
        "trade_count": len(tr),
        "prior_view_rate": round(matched / len(tr), 3),
    }


def position_evidence(snap):
    """BM453 -> daily holding-book series (replaces the BM451 weekly view).

    All metrics are computed locally from the daily snapshot so the harness
    owns the arithmetic instead of trusting a pre-aggregated upstream label.
    """
    rows = sorted(snap.get("account_daily") or [], key=lambda d: d.get("snapshot_date") or "")
    if not rows:
        return {"available": False}

    days = [d.get("snapshot_date") for d in rows]
    books = [(d.get("holdings") or {}) for d in rows]
    npos = [len(b) for b in books]
    stock_mv = [d.get("stock_total") for d in rows if d.get("stock_total") is not None]

    # per-ticker persistence + average weight inside the stock book
    appear, weight_sum = Counter(), defaultdict(float)
    for b in books:
        total = sum(float(p.get("stock_market_value") or 0) for p in b.values()) or 0.0
        for code, p in b.items():
            tk = (code or "").upper()
            appear[tk] += 1
            if total:
                weight_sum[tk] += float(p.get("stock_market_value") or 0) / total

    short_days = sum(1 for b in books if any((p.get("stock_vol") or 0) < 0 for p in b.values()))

    # book turnover: share of days whose holding set differs from the prior day
    changed = sum(1 for a, b in zip(books, books[1:]) if set(a) != set(b))
    transitions = max(len(books) - 1, 1)

    # unrealised P/L sign per day, aggregated over the whole book.
    # Uses last_price vs cost_price (the internally consistent basis) rather than
    # `profit_loss_total`: that field disagrees in MAGNITUDE with this basis on
    # 37.9% of rows, and the position engines all classify win/loss on the price
    # basis, so mixing them would let two parts of the profile disagree about
    # whether the same day was a losing one.
    pnl = []
    for b in books:
        if not b:
            continue
        s = 0.0
        for p in b.values():
            lp, cp = _f(p.get("last_price")), _f(p.get("cost_price"))
            if lp is None or cp is None or cp <= 0:
                continue
            s += (lp - cp) * (_f(p.get("stock_vol"), 0.0) or 0.0)
        pnl.append(s)

    # concentration of the stock book (top-1 holding market-value share)
    conc = []
    for b in books:
        total = sum(float(p.get("stock_market_value") or 0) for p in b.values()) or 0.0
        if total and b:
            top = max(float(p.get("stock_market_value") or 0) for p in b.values())
            conc.append(top / total)

    avg_npos = sum(npos) / len(npos) if npos else None
    diffs = [abs(b - a) for a, b in zip(npos, npos[1:])]
    churn = sum(diffs) / len(diffs) if diffs else 0.0
    # average day-over-day change in the number of holdings, relative to the
    # average book size: a book that drifts 9->10 is stable, one that swings
    # 0<->10 is not. Clipped to [0,1].
    stability = (round(max(0.0, 1 - churn / avg_npos), 3) if avg_npos
                 else (1.0 if npos else None))
    return {
        "available": True,
        "days": len(rows),
        "period_range": [days[0], days[-1]],
        "avg_num_positions": round(avg_npos, 2) if avg_npos else None,
        "position_count_stability": stability,
        "avg_daily_position_change": round(churn, 2),
        "max_num_positions": max(npos) if npos else 0,
        "days_with_shorts": short_days,
        "holding_turnover_rate": round(changed / transitions, 3),
        "avg_top1_concentration": round(sum(conc) / len(conc), 3) if conc else None,
        "avg_market_value_start": round(stock_mv[0], 2) if stock_mv else None,
        "avg_market_value_end": round(stock_mv[-1], 2) if stock_mv else None,
        "avg_market_value_change_pct": round((stock_mv[-1] - stock_mv[0]) / stock_mv[0], 4)
        if len(stock_mv) >= 2 and stock_mv[0] else None,
        "most_persistent_positions": [
            {"ticker": tk, "days_appear": c,
             "avg_weight_pct": round(weight_sum[tk] / c, 3) if c else None,
             **_ticker_info(tk)}
            for tk, c in appear.most_common(8)],
        "days_with_negative_unrealized_profit": sum(1 for v in pnl if v < 0),
        "days_with_positive_unrealized_profit": sum(1 for v in pnl if v > 0),
        "version": POSITION_EVIDENCE_VERSION,
    }


def asset_evidence(snap):
    """BM453 -> equity-curve metrics on a cash-flow-adjusted curve.

    The raw `asset_total` series moves for two reasons — the market moved, or the
    user added/withdrew money. Measuring return and drawdown on it therefore
    conflates performance with savings behaviour, which in this dataset is not
    theoretical: several users show raw "returns" of thousands of percent that
    are entirely deposits.

    So the headline numbers (`max_drawdown`, `equity_change_pct`,
    `equity_volatility`) are computed on a time-weighted curve with external
    flows removed, and the raw figures are kept alongside as `*_raw` plus a
    `flow_adjusted` flag. Field names are unchanged so downstream prompts, rules
    and the viewer stay stable.
    """
    rows = sorted(snap.get("account_daily") or [], key=lambda d: d.get("snapshot_date") or "")
    eq = [(d.get("snapshot_date"), _f(d.get("asset_total"))) for d in rows]
    eq = [(dt, float(v)) for dt, v in eq if v is not None]
    if not eq:
        return {"available": False}

    trades = snap.get("trades_6m") or []
    adj = adjusted_equity_series(rows, trades)
    dates, values = adj["dates"], adj["values"]
    if len(values) < 2:
        return {"available": False}

    raw_dates = [dt for dt, _ in eq]
    raw_values = [v for _, v in eq]

    dd, dd_date, peak_at, recovered = max_drawdown(dates, values)
    raw_dd, raw_dd_date, _, _ = max_drawdown(raw_dates, raw_values)
    rets = daily_returns(values)
    raw_rets = daily_returns(raw_values)

    flow_net = adj["flow_net"]
    first_raw = raw_values[0] if raw_values else 0
    material = bool(flow_net) and bool(first_raw) and abs(flow_net) > 0.05 * abs(first_raw)
    reliable = adj.get("reliable", True)

    return {
        "available": True,
        "date_range": [dates[0], dates[-1]],
        "num_days": len(values),
        "return_basis": ("insufficient" if not reliable else
                         "flow_adjusted" if adj["flow_count"] else "raw"),
        "return_reliable": reliable,
        "curve_coverage": adj.get("coverage"),
        "first_equity": round(values[0], 4),
        "last_equity": round(values[-1], 4),
        "equity_change_abs": round(values[-1] - values[0], 4),
        "equity_change_pct": total_return(values) if reliable else None,
        "max_drawdown": dd if reliable else None,
        "max_drawdown_date": dd_date if reliable else None,
        "recovered_by": recovered if reliable else None,
        "avg_equity": round(sum(values) / len(values), 4),
        # equity-curve family: `asset` is the single owner (it carries the full
        # raw audit trail). portfolio_metrics no longer recomputes these.
        "equity_volatility": round(statistics.pstdev(rets), 4) if len(rets) > 1 else None,
        "best_day_return": round(max(rets), 4) if rets else None,
        "worst_day_return": round(min(rets), 4) if rets else None,
        "unreliable_reason": (None if reliable else
                              f"Only {adj.get('coverage', 0) * 100:.0f}% of trading days are available for calculation (starting balance too small or abnormal single-day volatility), insufficient to provide return or drawdown conclusions"),
        # audit trail: what the naive calculation would have said
        "external_flow_events": adj["flow_count"],
        "external_flow_net": flow_net,
        "flow_material": material,
        "flow_adjusted": adj["flow_count"] > 0,
        "skipped_unreliable_days": adj["skipped_steps"],
        "equity_change_pct_raw": total_return(raw_values),
        "max_drawdown_raw": raw_dd,
        "max_drawdown_date_raw": raw_dd_date,
        "equity_volatility_raw": round(statistics.pstdev(raw_rets), 4) if len(raw_rets) > 1 else None,
        "version": ASSET_EVIDENCE_VERSION,
    }


_SELF_REF_RE = re.compile(r"(我|我的|my |mine|account|账户|持仓|portfolio|kyc|现金|了解我|对我)", re.I)
_PROMPT_LIKE_RE = re.compile(r"^(ask me anything|ai建议|prompt)", re.I)

_kw_hit = kw_hit
_ZH_RE = ZH_RE


def conversation_evidence(snap):
    """BM428 -> what the user actually asks about (conversation titles).

    The stream carries no timestamps, so no recency claim is made: the
    evidence is topical and self-referential only.
    """
    items = snap.get("conversations_1m") or []
    titles = [str(x.get("title") or "").strip() for x in items]
    titles = [t for t in titles if t]
    if not titles:
        return {"available": False}

    uniq = list(dict.fromkeys(titles))
    self_ref = [t for t in titles if _SELF_REF_RE.search(t)]

    tick = Counter()
    for t in titles:
        for tok in re.findall(r"\b[A-Z]{2,5}\b", t):
            if tok in SECTOR_MAP:
                tick[tok] += 1
        low = t.lower()
        for alias, tk in NAME_ALIASES.items():
            if alias in low:
                tick[tk] += 1

    themes = Counter()
    for tk, c in tick.items():
        themes[_ticker_info(tk)["theme"]] += c
    for theme, kws in TOPIC_KEYWORDS.items():
        hits = sum(1 for t in titles if any(_kw_hit(t.lower(), k) for k in kws))
        if hits:
            themes[theme] += hits

    return {
        "available": True,
        "count": len(titles),
        "unique_count": len(uniq),
        "dedup_ratio": round(len(uniq) / len(titles), 3),
        "self_reference_count": len(self_ref),
        "self_reference_share": round(len(self_ref) / len(titles), 3),
        "asked_about_own_portfolio": [t[:120] for t in self_ref[:6]],
        "tickers_mentioned": [{"ticker": tk, "count": c, **_ticker_info(tk)}
                              for tk, c in tick.most_common(10)],
        "themes": [{"theme": k, "count": c} for k, c in themes.most_common(8) if k != "Other"],
        "chinese_share": round(sum(1 for t in titles if _ZH_RE.search(t)) / len(titles), 3),
        "synthetic_prompt_like": sum(1 for t in titles if _PROMPT_LIKE_RE.match(t)),
        "sample_titles": uniq[:10],
    }


def _removed_stated_channel(snap):
    """REMOVED: `users.base_profile_summary` as a self-report channel.

    Kept as an explicit tombstone so the removal is visible to the next reader
    rather than looking like an omission. The field carried person-identifying
    data (name / nationality / address) that has no role in an investment
    profile: 国籍 94/157, 语言 96/157, 年龄 39/157 users, frequently explicit
    ("I am a 33-year-old Chinese national") and otherwise a negative ("has not
    provided details about…"). It was also not authentic self-report — an
    upstream LLM rewrote third-person facts, including a full
    `United States-California-Los Angeles` locality, into first person.

    Consumers had been: POLICY (all five dimensions, so the raw text reached
    every LLM prompt) and two rule fallbacks that used it for experience keywords
    and for a `stated_vs_revealed=aligned` claim which only checked that SOME text
    existed — not a comparison at all. See app/pii.py for the boundary guard.
    """
    return None


def trading_digest(tm, pm, asset=None):
    """Compact trading/portfolio view for the LLM prompt.

    The full metric blocks are large (hundreds of rows of holdings), so the
    proftlers get the headline figures only; the complete blocks are attached to
    the profile separately as computed output.

    `asset` supplies the equity-curve metrics (return / max drawdown /
    volatility). They used to be recomputed inside `portfolio_metrics.long_term`
    and reported a second time here in percent, so the same number reached the
    prompt twice under two names and two unit conventions. The equity curve now
    has exactly one owner, and this digest is its prompt-facing projection.
    """
    out = {}
    # The digest MUST always carry a top-level `available` flag: the permission
    # filter in profilers.run_profiler keeps only streams where available is
    # truthy, so omitting it silently hides this evidence from every agent —
    # which is exactly how the FIFO results (win rate, hold ratios, P&L
    # distribution) ended up invisible to the profiler that needs them most.
    # It is set explicitly here so "no data" is distinguishable from "forgot the
    # flag"; an absent key used to mean both.
    out["available"] = False
    if tm and tm.get("available"):
        a, e, r, b = tm.get("activity") or {}, tm.get("execution") or {}, tm.get("results") or {}, tm.get("behavior") or {}
        out["available"] = True
        out["trade"] = {
            "available": True, "timing_basis": tm.get("timing_basis"),
            "total_orders": a.get("total_orders"), "trading_days": a.get("trading_days"),
            "activity_rate": a.get("activity_rate"), "max_orders_per_day": a.get("max_orders_per_day"),
            "unique_tickers": a.get("unique_tickers"), "hhi_index": a.get("hhi_index"),
            "avg_hold_days": e.get("avg_hold_days"), "median_hold_days": e.get("median_hold_days"),
            "winner_hold_days": e.get("winner_hold_days"), "loser_hold_days": e.get("loser_hold_days"),
            "closed_trades": r.get("closed_trades"), "win_rate": r.get("win_rate"),
            "profit_factor": r.get("profit_factor"), "avg_win": r.get("avg_win"),
            "avg_loss": r.get("avg_loss"), "pnl_skewness": r.get("pnl_skewness"),
            "pnl_kurtosis": r.get("pnl_kurtosis"),
            "avg_down_hold_ratio": b.get("avg_down_hold_ratio"),
            "quick_profit_taking_ratio": b.get("quick_profit_taking_ratio"),
            "sample_note": (tm.get("coverage") or {}).get("note"),
        }
    if pm and pm.get("available"):
        lt, mt = pm.get("long_term") or {}, pm.get("medium_term") or {}
        rk = lt.get("risk_exposure") or {}
        # equity-curve metrics come from their single owner (`asset`), converted
        # to percent for the prompt. Nulls propagate honestly: when the curve is
        # unreliable the owner says so and nothing is invented here.
        a = asset or {}
        pct = (lambda v: round(v * 100, 2) if isinstance(v, (int, float)) else None)
        out["portfolio"] = {
            "available": True, "observation_days": pm.get("observation_days"),
            "total_return_pct": pct(a.get("equity_change_pct")),
            "total_return_pct_raw": pct(a.get("equity_change_pct_raw")),
            "return_basis": a.get("return_basis"),
            "max_drawdown_pct": pct(a.get("max_drawdown")),
            "volatility_pct": pct(a.get("equity_volatility")),
            "sharpe_ratio": rk.get("sharpe_ratio"),
            "avg_positions_4w": (mt.get("position_habit") or {}).get("avg_positions_4w"),
            "avg_leverage_4w": (mt.get("position_habit") or {}).get("avg_leverage_4w"),
            "monthly_turnover": (mt.get("medium_term_trading") or {}).get("monthly_turnover"),
            "fav_stocks_count": len((lt.get("preference") or {}).get("fav_stocks") or []),
            "fav_stocks_top": [x["ticker"] for x in ((lt.get("preference") or {}).get("fav_stocks") or [])[:5]],
            "avg_holding_weeks": (lt.get("holding_endurance") or {}).get("avg_holding_weeks"),
            "reliable": a.get("return_reliable"),
        }
    return out


def build_evidence(snap, behavior_intent=None):
    """Assemble the evidence package.

    `behavior_intent` lets the caller pass an already-computed BM433 engine
    result (the harness computes it during validation, so recomputing it here
    would double the per-user cost on multi-thousand-event users).
    """
    d = snap if isinstance(snap, dict) else snap.to_dict()
    if behavior_intent is None:
        behavior_intent = build_behavior_intent(d)
    trade_metrics = build_trade_metrics(d.get("trades_6m") or [])
    portfolio_metrics = build_portfolio_metrics(d)
    asset_ev = asset_evidence(d)
    # BM453 position dynamics and the BM433xBM453 cross engine. Both are large
    # (6-13KB each) so only the digests are exposed as evidence streams; the
    # full blocks ride along on the profile as computed output.
    position_dynamics = build_position_dynamics(d)
    attention_capital = build_attention_capital(d, behavior_intent)
    # The forward view. Built HERE, from the two engines above, rather than inside
    # the logic profiler: a forecast that cannot be recomputed deterministically is
    # not checkable, and the model must refine the engine's candidates rather than
    # invent its own (see profilers.DIM_HINT["logic"]).
    action_forecast = build_forecast(attention_capital, position_dynamics)
    return {
        "behavior": behavior_evidence(d),
        "behavior_intent": behavior_intent,
        "trade": trade_evidence(d),
        "trading_digest": trading_digest(trade_metrics, portfolio_metrics, asset_ev),
        "trade_metrics": trade_metrics,
        "portfolio_metrics": portfolio_metrics,
        "position_dynamics": position_dynamics,
        "position_dynamics_digest": position_dynamics_digest(position_dynamics),
        "attention_capital": attention_capital,
        "attention_capital_digest": attention_capital_digest(attention_capital),
        "action_forecast": action_forecast,
        "action_forecast_digest": forecast_digest(action_forecast),
        "behavior_trade_match": behavior_trade_matching(d),
        "position": position_evidence(d),
        "asset": asset_ev,
        "conversation": conversation_evidence(d),
    }
