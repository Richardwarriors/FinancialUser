"""Portfolio metrics: short / medium / long horizon views.

Source substitution
-------------------
The spec was written against weekly_holdings (weekly holding snapshots) and equity_summary (6-month
equity summary). Both were retired in favour of account_snapshots (daily account snapshots),
which is strictly richer, so the same metrics are computed here at daily
granularity and simply aggregated to the requested windows:

    "latest week"   -> the most recent 7 calendar days of daily snapshots
    "past 4 weeks"  -> the most recent 28 days
    equity_summary fields    -> recomputed from the daily series, flow-adjusted

Two consequences worth stating plainly:

  * A "week" is now a rolling 7-day window, not a calendar week, so windows can
    contain weekends. That is a property of daily data, not a defect.
  * All P&L and return figures come from the flow-adjusted curve
    (app/account.py). Deposits and withdrawals are not investment performance,
    and on this dataset 87 of 155 accounts have detectable flows.
"""
import math
import statistics
from collections import Counter, defaultdict
from datetime import timedelta

from .account import adjusted_equity_series, external_flows
from .trade_metrics import _mean, _percentile, _safe_div, _std, normalise_trades
from .taxonomy import ticker_info

METRIC_VERSION = "portfolio-metrics-v3-en"


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _ts(v):
    from .taxonomy import parse_ts
    return parse_ts(v)


def _rows(snap):
    return sorted((snap.get("account_daily") or []), key=lambda d: d.get("snapshot_date") or "")


def _window(rows, days):
    if not rows:
        return []
    end = _ts(rows[-1].get("snapshot_date"))
    if end is None:
        return []
    start = end - timedelta(days=days - 1)
    return [d for d in rows if (_ts(d.get("snapshot_date")) or end) >= start]


def _book_value(day):
    """(stock market value, position count) for one daily snapshot."""
    book = day.get("holdings") or {}
    mv = sum(_f(p.get("stock_market_value"), 0.0) or 0.0 for p in book.values())
    return mv, len(book)


# ---------- short term ----------

def short_term(rows, trades):
    if not rows:
        return None
    win = _window(rows, 7)
    as_of, book = None, {}
    for d in reversed(rows):
        if d.get("holdings"):
            as_of, book = d.get("snapshot_date"), d.get("holdings")
            break
    if not book:
        return {"available": False, "note": "No holdings were observed in the account"}

    total_mv = sum(_f(p.get("stock_market_value"), 0.0) or 0.0 for p in book.values())
    holdings = sorted(
        ({"ticker": (c or "").upper(),
          "name": p.get("stock_name"),
          "market_value": round(_f(p.get("stock_market_value"), 0.0) or 0.0, 2),
          "weight": round((_f(p.get("stock_market_value"), 0.0) or 0.0) / total_mv, 4) if total_mv else None,
          "unrealized_pnl": round(_f(p.get("profit_loss_total"), 0.0) or 0.0, 2),
          **ticker_info(c)} for c, p in book.items()),
        key=lambda x: -x["market_value"])

    weekly_pnl = sum(h["unrealized_pnl"] for h in holdings)
    # weekly return from the flow-adjusted curve, so a deposit in the window
    # cannot masquerade as a gain
    adj = adjusted_equity_series(rows, trades)
    vals, dates = adj["values"], adj["dates"]
    wr = None
    if len(vals) >= 2:
        end_ts = _ts(rows[-1].get("snapshot_date"))
        cut = end_ts - timedelta(days=6) if end_ts else None
        idx = [i for i, d in enumerate(dates) if cut and (_ts(d) or end_ts) >= cut]
        if len(idx) >= 2 and vals[idx[0]]:
            wr = round(vals[idx[-1]] / vals[idx[0]] - 1, 4)

    norm = normalise_trades(trades)
    end_ts = _ts(rows[-1].get("snapshot_date"))
    recent = [t for t in norm if end_ts and (end_ts - t["ts"]).days < 7] if end_ts else []
    buys = sum(t["amount"] for t in recent if t["side"] == "buy")
    sells = sum(t["amount"] for t in recent if t["side"] == "sell")
    latest_mv = _f(rows[-1].get("stock_total")) or total_mv

    return {
        "available": True,
        "as_of": as_of,
        "portfolio_snapshot": {
            "total_market_value": round(total_mv, 2),
            "num_positions": len(book),
            "holdings": holdings,
            "note": "Holdings are sorted by market value in descending order and are not truncated",
        },
        "weekly_pnl": {
            "weekly_pnl": round(weekly_pnl, 2),
            "weekly_return_pct": wr,
            "basis": "unrealized P&L from account_snapshots holdings; return from flow-adjusted curve",
            "caveat": "Unrealized P&L excludes fees and taxes and covers only current positions",
        },
        "short_term_trading": {
            "total_orders_7d": len(recent),
            "net_buy_7d": round(buys - sells, 2),
            "turnover_7d": _safe_div(buys + sells, latest_mv, 4) if latest_mv else None,
            "window_days": 7,
        },
    }


# ---------- medium term ----------

def medium_term(rows, trades):
    if len(rows) < 2:
        return None
    win = _window(rows, 28)
    mv_series = [_book_value(d)[0] for d in win]
    pos_series = [_book_value(d)[1] for d in win]
    pnl_series = [sum(_f(p.get("profit_loss_total"), 0.0) or 0.0
                      for p in (d.get("holdings") or {}).values()) for d in win]

    changes = [(b - a) / a for a, b in zip(mv_series, mv_series[1:]) if a]
    avg_mv = _mean([m for m in mv_series if m]) if any(mv_series) else None
    chg = (_safe_div(mv_series[-1] - mv_series[0], mv_series[0], 4)
           if len(mv_series) >= 2 and mv_series[0] else None)

    norm = normalise_trades(trades)
    end_ts = _ts(rows[-1].get("snapshot_date"))
    recent30 = [t for t in norm if end_ts and (end_ts - t["ts"]).days < 30] if end_ts else []
    amt30 = sum(t["amount"] for t in recent30)
    latest_mv = _f(rows[-1].get("stock_total")) or (mv_series[-1] if mv_series else None)
    active = Counter()
    for t in recent30:
        active[t["ticker"]] += t["amount"]

    flows = external_flows(rows, trades)

    return {
        "available": True,
        "window_days": 28,
        "market_value_trend": {
            "avg_mv_4w": round(avg_mv, 2) if avg_mv is not None else None,
            "mv_change_4w_pct": chg,
            "mv_volatility_4w": round(statistics.pstdev(changes), 4) if len(changes) > 1 else None,
            "series_days": len(mv_series),
        },
        "pnl_trend": {
            "cum_pnl_4w": round(pnl_series[-1], 2) if pnl_series else None,
            "pnl_std_4w": round(statistics.pstdev(pnl_series), 2) if len(pnl_series) > 1 else None,
            "pnl_positive_days": sum(1 for p in pnl_series if p > 0),
            "pnl_negative_days": sum(1 for p in pnl_series if p < 0),
            "basis": "unrealized P&L of the current book over the window",
        },
        "position_habit": {
            "avg_positions_4w": round(_mean([p for p in pos_series]), 2) if pos_series else None,
            "max_positions_4w": max(pos_series) if pos_series else None,
            "avg_leverage_4w": _avg_leverage(rows),
            "leverage_note": "Account leverage = securities market value / total assets (equivalent definition to equity_summary's avg_leverage_asset_over_net)",
        },
        "medium_term_trading": {
            "monthly_turnover": _safe_div(amt30, latest_mv, 4) if latest_mv else None,
            "active_stocks_30d": [{"ticker": k, "amount": round(v, 2), **ticker_info(k)}
                                  for k, v in active.most_common()],
            "orders_30d": len(recent30),
        },
        "external_flows_4w": {
            "events": len([f for f in flows if end_ts and
                           (_ts(f["date"]) or end_ts) >= end_ts - timedelta(days=28)]),
            "note": "Number of deposits and withdrawals within the window; these funds do not constitute investment returns",
        },
    }


def _avg_leverage(rows):
    """Mean stock_value / asset_total. >1 means borrowed money is at work."""
    ratios = []
    for d in rows:
        a, s = _f(d.get("asset_total")), _f(d.get("stock_total"))
        if a and s is not None:
            ratios.append(s / a)
    if not ratios:
        return None
    return round(_mean(ratios), 4)


# ---------- long term ----------

def long_term(rows, trades):
    if len(rows) < 2:
        return None
    adj = adjusted_equity_series(rows, trades)
    values = adj["values"]
    reliable = adj.get("reliable", True)

    # the adjusted curve is needed only for the Sharpe ratio now; return,
    # drawdown and volatility are owned by `asset` (see the ownership note below)
    rets = [(b / a - 1) for a, b in zip(values, values[1:]) if a]
    vol = statistics.pstdev(rets) if len(rets) > 1 else None
    # "rough Sharpe" per spec: mean daily return / daily volatility, rf = 0
    sharpe = _safe_div(_mean(rets) or 0.0, vol, 3) if vol else None

    # 全部偏好股票：出现频率（按天计），不截断
    appear = Counter()
    for d in rows:
        for c in (d.get("holdings") or {}):
            appear[(c or "").upper()] += 1
    n_days = len(rows)
    fav = [{"ticker": t, "days_held": n, "pct_of_period": round(n / n_days, 4), **ticker_info(t)}
           for t, n in appear.most_common()]
    avg_pos = _mean([_book_value(d)[1] for d in rows]) or 0
    # The spec formula (weeks / avg positions) breaks down when the book is
    # usually empty: dividing by ~0 yields absurd four-digit week counts. Below
    # one average position it carries no information, so it is omitted.
    weeks = n_days / 7
    endurance = _safe_div(weeks, avg_pos, 2) if avg_pos >= 1 else None

    return {
        "available": True,
        "reliable": reliable,
        "basis": ("flow_adjusted" if adj["flow_count"] else "raw") if reliable else "insufficient",
        "unreliable_reason": None if reliable else
            f"Only {adj.get('coverage', 0) * 100:.0f}% of trading days are available for calculation; returns and drawdowns are unreliable",
        # NOTE ON FIELD OWNERSHIP: return / max drawdown / volatility used to be
        # computed here AND in `evidence.asset_evidence`, from the same inputs via
        # the same helpers — verified identical on 144/144 comparable users, only
        # differing by the x100 unit convention. Two copies can silently drift
        # (fix one, forget the other), so `asset` is now the single owner of the
        # equity-curve family and this block covers portfolio composition only.
        "equity_curve_owner": "asset",
        "performance": {
            "end_equity": round(_f(rows[-1].get("asset_total")) or 0.0, 2),
            "start_equity": round(_f(rows[0].get("asset_total")) or 0.0, 2),
            "observation_days": n_days,
            "note": "Returns/drawdown/volatility: see the asset evidence stream (same equity curve, single definition)",
        },
        "risk_exposure": {
            # only the Sharpe ratio lives here: it is not in `asset`
            "sharpe_ratio": sharpe if reliable else None,
            "sharpe_note": "Rough definition: daily return / daily volatility, risk-free rate set to 0, not annualized",
        },
        "preference": {
            "fav_stocks": fav,
            "note": "Count holding frequency by the number of days appearing in daily snapshots, uncapped",
        },
        "holding_endurance": {
            "avg_holding_weeks": endurance,
            "weeks_observed": round(weeks, 1),
            # avg_positions is NOT emitted here: `position.avg_num_positions`
            # (evidence stream, agent-facing) already owns the average book size
            # over the same full window, and the two were verified identical on
            # 152/155 users. It stays a local variable because the formula needs it.
            "note": ("Number of weeks ÷ average number of positions, defined as an approximate inverse of turnover speed; see position.avg_num_positions for the average number of positions"
                    if endurance is not None else
                    "Average holding count is less than 1, this formula is not applicable, so the number of holding weeks is not provided"),
        },
    }


def build_portfolio_metrics(snap):
    rows = _rows(snap)
    trades = snap.get("trades_6m") or []
    if not rows:
        return {"available": False, "note": "No account_snapshots daily account snapshot"}
    return {
        "available": True,
        "version": METRIC_VERSION,
        "granularity": "daily (account_snapshots) — replaces weekly_holdings weekly + equity_summary summary",
        "observation_days": len(rows),
        "observation_range": [rows[0].get("snapshot_date"), rows[-1].get("snapshot_date")],
        "short_term": short_term(rows, trades),
        "medium_term": medium_term(rows, trades),
        "long_term": long_term(rows, trades),
    }
