"""Attention x Capital alignment: BM433 (attention) x BM453 (capital).

The premise
-----------
BM433 says where the user's **attention** goes; BM453 says where their **money**
goes. Crossing them answers questions neither stream can answer alone: is this
a monitored conviction holding, a passive core position, an exploration that
never converted, or capital committed without any monitoring?

Reach
-----
152 of 157 users have both streams (against only 7 with BM452 trades), so this
is the first cross-source analysis in the project that works at corpus scale.

Honesty rules applied throughout
--------------------------------
* Day-1 holdings are **not** entries. A ticker present in the first snapshot was
  already owned before the window; we cannot see when it was bought, so it is
  marked `held_at_start` and excluded from every entry statistic.
* A conversion rate needs a real denominator. "Viewed 100 tickers, holds 15" is
  not a conversion rate — the 15 may predate the window. Conversion is only
  computed over tickers that were **not held at window start**.
* Ratios over thin denominators are refused, not reported. Every rate carries
  the counts it was built from.
* Attention is a share of observed behaviour, not a psychological quantity. All
  output is behavioural description; nothing here is a recommendation.
"""
from collections import Counter, defaultdict
from datetime import timedelta

from .account import adjusted_equity_series
from .behavior import INTENT_STAGES, _STAGE_RANK, build_sessions, normalize_stream
from .position_events import baseline_index, build_days, position_state
from .taxonomy import ticker_info
ATTENTION_CAPITAL_VERSION = "attention-capital-v2"

MIN_ENTRIES_FOR_RATE = 3
# An entry *count* of 3 is informative; a *rate* over 3 samples is not. A rate
# needs its own, larger floor or a 3-ticker universe reports a confident "1.0".
MIN_DENOM_FOR_RATE = 8          # below this a conversion rate is noise
MIN_DAYS_FOR_LAG = 20             # below this lead-lag is meaningless
ZERO_POSITION_EPS = 1e-9


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_div(a, b, digits=4):
    if not b:
        return None
    return round(a / b, digits)


def _ts(v):
    from .taxonomy import parse_ts
    return parse_ts(v)


# ---------- capital side ----------

def capital_timeline(gated_days):
    """Per-day holdings with market value, weight and P&L.

    Takes days that have already passed the data-quality gate
    (position_events.build_days). Reading the raw snapshots here would let a
    `holdings = {} while stock_total > 0` gap day look like a full liquidation —
    and because day 1 of *every* user examined is such a day, that mistake
    turned the entire second day of holdings into ~57 phantom "entries".

    Returns (days, weights, values, pnl, states).

    `pnl` is `profit_loss_total`, kept for reference only: it disagrees with
    last_price vs cost_price on 37.9% of rows by magnitude, so it is never
    aggregated and never used to classify win/loss. `states` carries the
    internally consistent verdict (gain/loss/flat/unknown) instead.
    """
    days, weights, values, pnl, states = [], [], [], [], []
    for d in gated_days:
        book = d.get("holdings") or {}
        total = sum(_f(p.get("stock_market_value")) for p in book.values())
        w, v, g, s = {}, {}, {}, {}
        for c, p in book.items():
            t = (c or "").upper()
            mv = _f(p.get("stock_market_value"))
            v[t] = mv
            w[t] = (mv / total) if total else 0.0
            g[t] = _f(p.get("profit_loss_total"))
            state, _ = position_state(p)
            s[t] = state
        days.append(d.get("date"))
        weights.append(w)
        values.append(v)
        pnl.append(g)
        states.append(s)
    return days, weights, values, pnl, states


def _ticker_lifecycle(days, values, baseline=0):
    """First/last date each ticker appears, and whether it was there at the open.

    `held_at_start` matters: those tickers were bought before the observation
    window, so treating their first appearance as an entry invents fake
    "entries". `baseline` is the first day with a *known* book (see
    position_events.baseline_index) — day 0 is usually an unknown book, so
    anchoring on index 0 would mark nobody as pre-held.
    """
    first = {}
    for i, d in enumerate(days):
        for t in values[i]:
            first.setdefault(t, i)
    start = days[baseline] if days and baseline < len(days) else (days[0] if days else None)
    out = {}
    for t, i in first.items():
        out[t] = {"first_seen": days[i], "first_index": i,
                  "held_at_start": i <= baseline, "last_seen": None,
                  "days_held": sum(1 for v in values if t in v)}
    for i, d in enumerate(days):
        for t in values[i]:
            out[t]["last_seen"] = d
    return out


def entries_and_exits(days, values, lifecycle, baseline=0, known=None):
    """Observable position events, excluding anything already held at the open.

    Unknown-book days are skipped and do not move the baseline forward, so a
    data gap is never reported as a liquidation followed by a re-entry.
    """
    ok = _known_mask(days, known)
    entries, exits = [], []
    prev_idx = baseline
    prev = set(values[baseline]) if baseline < len(values) else set()
    for i, d in enumerate(days):
        if i <= baseline or not ok[i]:
            continue                          # already owned, or book unknown
        cur = set(values[i])
        for t in cur - prev:
            entries.append({"ticker": t, "date": d, "index": i})
        for t in prev - cur:
            exits.append({"ticker": t, "date": d, "index": i,
                          "days_held": lifecycle.get(t, {}).get("days_held")})
        prev = cur
        prev_idx = i
    return entries, exits


def _known_mask(days, known):
    """Boolean per day: is the holding book trustworthy enough to diff on?

    Days whose book is unknown must be skipped *without* updating the previous
    book. Otherwise an empty unknown day looks like a full liquidation and the
    next day looks like a wave of fresh entries.
    """
    if known is None:
        return [True] * len(days)
    return [bool(k) for k in known]


def scale_events(days, values, weights, known=None):
    """Weight increases/decreases on an already-held position.

    A share change is only counted when it is material (>20% relative move) so
    that price drift does not masquerade as a decision to resize.
    """
    ok = _known_mask(days, known)
    ups, downs = [], []
    for i in range(1, len(days)):
        if not (ok[i] and ok[i - 1]):
            continue
        for t, w in weights[i].items():
            if t not in weights[i - 1] or t not in values[i - 1]:
                continue
            prev_v, cur_v = values[i - 1][t], values[i][t]
            if prev_v <= 0:
                continue
            chg = cur_v / prev_v - 1
            if chg >= 0.2:
                ups.append({"ticker": t, "date": days[i], "index": i, "change": round(chg, 3)})
            elif chg <= -0.2:
                downs.append({"ticker": t, "date": days[i], "index": i, "change": round(chg, 3)})
    return ups, downs


# ---------- attention side ----------

def attention_by_ticker(events):
    """Full per-ticker attention counts (not top-N like the digest)."""
    cnt = Counter()
    for e in events:
        t = e.get("focus_ticker")
        if t:
            cnt[t] += 1
    return cnt


def attention_by_day_ticker(events):
    by = defaultdict(Counter)
    for e in events:
        t = e.get("focus_ticker")
        if t:
            by[e["date"]][t] += 1
    return by


def _attention_in_window(events, ticker, end_ts, days):
    start = end_ts - timedelta(days=days)
    return [e for e in events
            if (e.get("focus_ticker") == ticker) and start < e["ts"] <= end_ts]


def _theme_share(weights_row):
    th = defaultdict(float)
    for t, w in weights_row.items():
        th[ticker_info(t)["theme"]] += w
    return th


# ---------- feature blocks ----------

def orientation(events, attention, held_set):
    """Monitoring vs exploration: where does attention actually point?"""
    total = sum(attention.values())
    if not total:
        return None
    held = sum(c for t, c in attention.items() if t in held_set)
    non = total - held
    return {
        "available": True,
        "attention_on_held": held,
        "attention_on_non_held": non,
        "held_ticker_view_ratio": _safe_div(held, total, 3),
        "nonheld_ticker_view_ratio": _safe_div(non, total, 3),
        "held_tickers_attended": len([t for t in attention if t in held_set]),
        "nonheld_tickers_attended": len([t for t in attention if t not in held_set]),
        "interpretation": ("Primarily focused on monitoring existing positions (Portfolio Monitor)" if held > non * 1.5 else
                           "Primarily focused on finding new opportunities (Opportunity Explorer)" if non > held * 1.5 else
                           "Balances monitoring holdings and exploring new instruments"),
        "note": "Numerator: number of behavioral events falling on the latest holdings; denominator: total number of behavioral events attributable to instruments",
    }


def alignment(attention, latest_weights, top_n=12):
    """attention_share - position_weight per ticker, with the 2x2 state.

    Quadrants (attention share 5%, position weight 5% / 15%):
      high attention + held      -> active_conviction   (watching what you own)
      high attention + not held  -> potential_entry     (watched, unowned)
      low attention  + heavy     -> passive_core_holding (big, ignored)
      low attention  + light     -> minor_holding       (small, ignored)

    `minor_holding` used to be called `low_relevance`, which was actively
    misleading: a 12% position is not irrelevant, it is simply small and
    unmonitored, and as a dominant_state the old name told the reader the whole
    book did not matter.
    """
    total_att = sum(attention.values())
    if not total_att:
        return None
    rows = []
    for t, w in latest_weights.items():
        a = attention.get(t, 0) / total_att
        gap = a - w
        if a >= 0.05 and w >= 0.05:
            state = "high_conviction_monitoring"
        elif a >= 0.05 and w < 0.05:
            state = "potential_entry"
        elif a < 0.05 and w >= 0.15:
            state = "passive_core_holding"
        else:
            state = "minor_holding"
        rows.append({"ticker": t, "attention_share": round(a, 4),
                     "position_weight": round(w, 4), "attention_capital_gap": round(gap, 4),
                     "state": state, **ticker_info(t)})
    for t, c in attention.items():
        if t in latest_weights:
            continue
        a = c / total_att
        if a >= 0.03:
            rows.append({"ticker": t, "attention_share": round(a, 4), "position_weight": 0.0,
                         "attention_capital_gap": round(a, 4),
                         "state": "potential_entry", **ticker_info(t)})
    rows.sort(key=lambda r: -abs(r["attention_capital_gap"]))
    # which tickers are actually held (weight > 0) vs merely watched, for state labels
    held_rows = [r for r in rows if r["position_weight"] > 0]
    return {
        "available": True,
        "attention_total_events": total_att,
        "largest_gaps": rows[:top_n],
        "attention_gt_capital": [r["ticker"] for r in rows if r["attention_capital_gap"] > 0.03][:8],
        "capital_gt_attention": [r["ticker"] for r in rows if r["attention_capital_gap"] < -0.03][:8],
        "state_counts": dict(Counter(r["state"] for r in rows)),
        # dominant_state names the single most common relationship, so the reader
        # does not have to take an argmax over state_counts themselves
        "dominant_state": (Counter(r["state"] for r in held_rows).most_common(1)[0][0]
                           if held_rows else None),
        "attention_only_tickers": [r["ticker"] for r in rows if r["position_weight"] == 0][:10],
        "note": ("gap>0 indicates more attention than allocation (potential interest in starting or adding to a position), while gap<0 indicates more allocation than attention (passive core holding). dominant_state is calculated among held instruments, and attention_only_tickers are instruments that were never held but were continuously followed during the period."),
    }


def pre_entry_pattern(events, entries):
    """Attention accumulated before each newly observed position."""
    if not entries:
        return {"available": False,
                "note": "No identifiable new position starts during the observation window (first-day holdings excluded)"}
    res = []
    for en in entries:
        end = _ts(en["date"] + "T23:59")
        row = {"ticker": en["ticker"], "date": en["date"]}
        for d in (1, 3, 7, 30):
            w = _attention_in_window(events, en["ticker"], end, d)
            row[f"pre_entry_attention_{d}d"] = len(w)
            if d == 7:
                stages = [e["intent_stage"] for e in w]
                row["pre_entry_research_depth"] = (
                    max(stages, key=lambda s: _STAGE_RANK[s]) if stages else None)
                row["pre_entry_active_days"] = len({e["date"] for e in w})
                types = Counter(e["event_type"] for e in w)
                row["pre_entry_top_actions"] = [k for k, _ in types.most_common(3)]
        res.append(row)
    with7 = [r for r in res if r["pre_entry_attention_7d"] > 0]
    depths = Counter(r.get("pre_entry_research_depth") for r in res)
    return {
        "available": True,
        "entries": len(res),
        "entries_with_prior_attention_7d": len(with7),
        "prior_attention_rate_7d": _safe_div(len(with7), len(res), 3),
        "pre_entry_research_depth_mix": {k: v for k, v in depths.items() if k},
        "median_pre_entry_attention_7d": (sorted(r["pre_entry_attention_7d"] for r in res)[len(res) // 2]
                                          if res else None),
        "examples": res[:10],
        "note": "The window ends at 23:59 on the position-opening date (pre-market/intraday research on that day is included); first-day holdings are not counted as position opening",
    }


def conversion(events, lifecycle, entries, weights, days, research_by_ticker):
    """Research -> holding conversion, with a defensible denominator."""
    not_held_at_start = {t for t, v in lifecycle.items() if not v["held_at_start"]}
    attended_not_held = {t: c for t, c in research_by_ticker.items()
                         if t not in lifecycle or not lifecycle[t]["held_at_start"]}
    entered = {e["ticker"] for e in entries} & not_held_at_start
    deep = {t for t, stage in research_by_ticker.items()
            if _STAGE_RANK.get(stage, 0) >= _STAGE_RANK["research"]}
    deep_not_held = deep & set(attended_not_held) & not_held_at_start

    out = {
        "available": True,
        "universe_attended_not_held": len(attended_not_held),
        "entries_observed": len(entries),
        "note": ("The denominator includes only instruments not held at the start of the observation window; positions held on the first day cannot be used to determine the position-building time and have been excluded. Research level is determined by the highest intent stage reached for the instrument."),
    }
    if len(attended_not_held) >= MIN_DENOM_FOR_RATE:
        out["research_to_holding_conversion"] = _safe_div(len(entered), len(attended_not_held), 3)
        out["research_to_holding_denominator"] = len(attended_not_held)
        out["research_to_holding_numerator"] = len(entered)
    else:
        out["research_to_holding_note"] = (
            f"Only {len(attended_not_held)} watched instruments were not held during the observation window, fewer than {MIN_DENOM_FOR_RATE}; no conversion rate provided (sample too small)")
    if len(deep_not_held) >= MIN_DENOM_FOR_RATE:
        out["deep_research_to_holding_conversion"] = _safe_div(
            len(deep_not_held & entered), len(deep_not_held), 3)
        out["deep_research_denominator"] = len(deep_not_held)
    else:
        out["deep_research_note"] = (
            f"Only {len(deep_not_held)} research-grade non-held instruments were identified, fewer than {MIN_DENOM_FOR_RATE}, so no deep research conversion rate is provided")
    return out


def capital_conversion(events, entries, weights, days):
    """E[future position weight | research depth at entry]."""
    if not entries or len(days) < 5:
        return {"available": False, "note": "Insufficient position-opening samples to estimate the relationship between attention intensity and subsequent position size"}
    buckets = defaultdict(list)
    for en in entries:
        end = _ts(en["date"] + "T23:59")
        w7 = _attention_in_window(events, en["ticker"], end, 7)
        stages = [e["intent_stage"] for e in w7]
        depth = max(stages, key=lambda s: _STAGE_RANK[s]) if stages else "none"
        # forward weight: max weight observed in the 30 days after entry
        fwd = 0.0
        for i in range(en["index"], min(en["index"] + 30, len(weights))):
            fwd = max(fwd, weights[i].get(en["ticker"], 0.0))
        buckets[depth].append(fwd)
    shown = {}
    for k, v in buckets.items():
        if len(v) < 2:
            continue
        shown[k] = {"entries": len(v), "avg_forward_weight": round(sum(v) / len(v), 4),
                    "max_forward_weight": round(max(v), 4)}
    return {
        "available": bool(shown),
        "by_research_depth_at_entry": shown,
        "note": ("The forward position is the highest weight within 30 days after position building. Use for reference only when the sample is very small; it should not be treated as a causal conclusion.") if shown else "Fewer than 2 samples in each bucket; statistics not provided",
        "basis": "max weight within 30 days after the observed entry",
    }


def persistence_pairing(events, lifecycle, days):
    """Attention persistence x holding persistence -> relationship type."""
    span = max(len(days), 1)
    att_days = defaultdict(set)
    for e in events:
        t = e.get("focus_ticker")
        if t:
            att_days[t].add(e["date"])
    rows = []
    for t, info in lifecycle.items():
        a = len(att_days.get(t, ()))
        h = info["days_held"]
        att_persist = a / span
        hold_persist = h / span
        if att_persist >= 0.15 and hold_persist >= 0.5:
            kind = "core_conviction"
        elif att_persist >= 0.15 and hold_persist < 0.5:
            kind = "active_trading_or_speculation"
        elif att_persist < 0.15 and hold_persist >= 0.5:
            kind = "passive_core_holding"
        else:
            kind = "episodic_interest"
        rows.append({"ticker": t, "attention_days": a, "days_held": h,
                     "attention_persistence": round(att_persist, 3),
                     "holding_persistence": round(hold_persist, 3),
                     "relationship": kind, **ticker_info(t)})
    rows.sort(key=lambda r: -(r["attention_days"] + r["days_held"]))
    return {
        "available": True,
        "relationship_mix": dict(Counter(r["relationship"] for r in rows)),
        "tickers": rows[:12],
        "note": "Attention persistence = proportion of trading days with activity; holding persistence = proportion of trading days appearing in holdings",
    }


def concentration_compare(attention, latest_weights):
    """Attention HHI vs portfolio HHI -> focus style."""
    ta = sum(attention.values())
    tw = sum(latest_weights.values())
    if not ta or not tw:
        return None
    a_hhi = sum((c / ta) ** 2 for c in attention.values())
    w_hhi = sum((w / tw) ** 2 for w in latest_weights.values())
    if a_hhi >= 0.15 and w_hhi >= 0.25:
        style = "Highly concentrated (focused in both research and capital)"
    elif a_hhi >= 0.15 and w_hhi < 0.25:
        style = "Research is concentrated but capital is diversified"
    elif a_hhi < 0.15 and w_hhi >= 0.25:
        style = "Broad exploration but concentrated capital"
    else:
        style = "Research and capital are relatively diversified"
    return {
        "available": True,
        "attention_hhi": round(a_hhi, 4),
        "portfolio_hhi": round(w_hhi, 4),
        "style": style,
        "note": ("The higher the HHI, the greater the concentration. Attention HHI is based on the distribution of behavior events, while portfolio HHI is based on the market value distribution of holdings. The two use different definitions and are for style comparison only."),
    }


def loss_conditioned(events, days, values, states):
    """Does attention rise when a position is underwater?

    Win/loss comes from last_price vs cost_price (via position_state), not from
    profit_loss_total, whose magnitude disagrees with that basis on 37.9% of rows.

    Each event is attributed using the position state ON THE EVENT'S OWN DATE.
    Aggregating a ticker's day counts first and then labelling all of its events
    by the majority state collapsed the whole signal: a ticker down 60 of 90 days
    sent 100% of its attention to the "losing" bucket, producing 1213-vs-0 for a
    portfolio that was up over the window.

    Evidence: an event only counts if the ticker was actually held that day, so
    attention paid to a ticker before it was ever bought is not evidence about
    how the user reacts to gains or losses.
    """
    state_on_day = {days[i]: states[i] for i in range(len(days))}
    lose_att = win_att = 0
    unattributable = 0
    for e in events:
        t = e.get("focus_ticker")
        if not t:
            continue
        st = (state_on_day.get(e.get("date")) or {}).get(t)
        if st == "loss":
            lose_att += 1
        elif st == "gain":
            win_att += 1
        else:
            unattributable += 1
    total = lose_att + win_att
    if not total:
        return {"available": False, "note": "No behavioral events attributable to a position P&L state"}
    share = _safe_div(lose_att, total, 3)
    ratio = _safe_div(lose_att, win_att, 2)
    # A 2-event sample must not produce a directional claim. Report the counts,
    # withhold the interpretation.
    if total < MIN_DENOM_FOR_RATE:
        return {
            "available": True,
            "attention_on_losing_positions": lose_att,
            "attention_on_winning_positions": win_att,
            "attention_unattributable_events": unattributable,
            "loss_attention_ratio": ratio,
            "share_on_losing": share,
            "interpretation": None,
            "sample_note": (f"Only {total} attributable events, fewer than {MIN_DENOM_FOR_RATE}; the sample is insufficient to determine whether attention is biased toward profit or loss states"),
            "limitation": ("Only counts events where the instrument was actually held that day; events on days when it was not held are excluded."),
        }
    # When nothing was watched while in profit, the ratio is undefined but the
    # situation is informative, not neutral: every attributable event happened
    # while underwater. Read the share, not the (absent) ratio.
    if win_att == 0:
        interpretation = "All attributable attention occurred while in a loss state (no profitable-state samples, unable to calculate the ratio)"
    elif lose_att == 0:
        interpretation = "All attributable attention occurred while in a profitable state"
    elif ratio is not None and ratio > 1.5:
        interpretation = "Shows significantly more attention during losses (possibly anxiety, averaging-down research, or stop-loss evaluation)"
    elif ratio is not None and ratio < 0.67:
        interpretation = "Pays more attention when profitable (possibly momentum tracking or profit-taking monitoring)"
    else:
        interpretation = "No significant attention bias under profit or loss conditions"
    return {
        "available": True,
        "attention_on_losing_positions": lose_att,
        "attention_on_winning_positions": win_att,
        "attention_unattributable_events": unattributable,
        "loss_attention_ratio": ratio,
        "share_on_losing": share,
        "interpretation": interpretation,
        "limitation": ("Can only establish that attention and an unrealized loss occurred simultaneously; causation cannot be determined. Motivation must be inferred from subsequent add/trim behavior (see scale_events). Counts only events on days when the instrument was actually held; events on days when it was not held are excluded."),
    }


MIN_OPPORTUNITIES = 20       # below this, a conditional rate is noise
ATTENTION_LOOKBACK_DAYS = 7  # "recently looked" = strictly before the action
# A denominator is only half the story. With 20+ opportunity days but ONE add, the
# "rate" is 1/122 = 0.8% and the comparison is between single acts, not tendencies:
# the first version was gated on opportunities alone and told 22 users that looking
# "more often" precedes adding, most of them on 1-2 adds. Same rule as
# MIN_EPISODES_FOR_STYLE: one act is not a tendency.
MIN_COMPARISON_ADDS = 3


def attention_to_action(events, days, states, weights, values, known=None,
                        lookback=ATTENTION_LOOKBACK_DAYS):
    """Does attention PRECEDE action? The conditional chain the ratio cannot give.

    `loss_attention_ratio` says attention and being underwater co-occur. It cannot
    distinguish three very different users who all score ~1.0: the one who watches a
    loser and does nothing, the one who watches and adds, and the one who ignores it
    and adds anyway. This closes that gap by measuring, per day the position was
    underwater, whether an add followed — split by whether the user had looked at
    that ticker in the preceding week.

    Unit of observation is a **(ticker, day) opportunity** where the position was
    held and underwater, so an add was possible. That gives a real denominator
    instead of a handful of events.

    Same-day attention does NOT count as preceding: it cannot be told apart from
    looking after the decision was already made. Only days strictly before the
    action are counted, which is why the lookback is stated on the payload.
    """
    state_on_day = {days[i]: states[i] for i in range(len(days))}
    ok = _known_mask(days, known)

    # attention by (ticker, date) so a day can be tested directly
    att_by_day = Counter()
    for e in events:
        t = e.get("focus_ticker")
        if t and e.get("date"):
            att_by_day[(t, e["date"])] += 1

    # idx[date] -> position, for the lookback window
    day_pos = {d: i for i, d in enumerate(days)}

    def had_recent(ticker, date):
        i = day_pos.get(date)
        if i is None:
            return False
        for j in range(max(0, i - lookback), i):      # strictly before
            if att_by_day.get((ticker, days[j])):
                return True
        return False

    buckets = {
        "loss_with_attention": {"opportunities": 0, "adds": 0, "reduces": 0},
        "loss_without_attention": {"opportunities": 0, "adds": 0, "reduces": 0},
        "gain_with_attention": {"opportunities": 0, "adds": 0, "reduces": 0},
        "gain_without_attention": {"opportunities": 0, "adds": 0, "reduces": 0},
    }
    examples = []
    for i in range(1, len(days)):
        if not ok[i] or not ok[i - 1]:
            continue
        for t in weights[i]:
            # an add requires the position to have existed the day before
            if t not in weights[i - 1] or t not in values[i - 1] or values[i - 1][t] <= 0:
                continue
            st = (states[i] or {}).get(t)
            if st not in ("loss", "gain"):
                continue
            chg = values[i][t] / values[i - 1][t] - 1
            looked = had_recent(t, days[i])
            key = f"{st}_{'with' if looked else 'without'}_attention"
            b = buckets[key]
            b["opportunities"] += 1
            if chg >= 0.2:                      # same materiality rule as scale_events
                b["adds"] += 1
                if st == "loss" and len(examples) < 12:
                    examples.append({"ticker": t, "date": days[i],
                                     "state": st, "change": round(chg, 3),
                                     "recent_attention": looked})
            elif chg <= -0.2:
                b["reduces"] += 1

    def rate(b, field="adds"):
        return (round(b[field] / b["opportunities"], 3)
                if b["opportunities"] >= MIN_OPPORTUNITIES else None)

    loss_with = buckets["loss_with_attention"]
    loss_without = buckets["loss_without_attention"]
    r_with, r_without = rate(loss_with), rate(loss_without)
    n_adds = loss_with["adds"] + loss_without["adds"]

    if r_with is None or r_without is None:
        gate = "opportunities_insufficient"
        interpretation = (f"Insufficient loss-period opportunity-day sample (after viewing {loss_with['opportunities']} days, not viewed {loss_without['opportunities']} days, each requiring ≥{MIN_OPPORTUNITIES} days); no conditional ratio")
    elif n_adds < MIN_COMPARISON_ADDS:
        # The denominator is fine; the NUMERATOR is not. Two buckets holding one act
        # between them cannot show which situation "more often" leads to adding.
        gate = "adds_insufficient"
        interpretation = (f"Too few additions during the loss period (after viewing {loss_with['adds']} times, not viewed {loss_without['adds']} times, {n_adds} times in total, fewer than {MIN_COMPARISON_ADDS} times), insufficient to compare the tendency to add after viewing versus without viewing. Enough opportunity days, but not enough events.")
    elif r_with > r_without * 1.5:
        gate = "ok"
        interpretation = (f'When unrealized losses occur, "look before acting" is more common: add-position rate on days viewed {r_with:.1%}, on days not viewed {r_without:.1%}')
    elif r_without > r_with * 1.5:
        gate = "ok"
        interpretation = (f"Adding to positions while at a loss does not depend on recent attention: add-to-position rate on viewed days {r_with:.1%}, not viewed {r_without:.1%}")
    else:
        gate = "ok"
        interpretation = (f"When unrealized losses occur, add-position activity is not clearly related to whether the instrument was recently viewed (viewed {r_with:.1%}, not viewed {r_without:.1%})")

    return {
        "available": True,
        "lookback_days": lookback,
        "min_opportunities": MIN_OPPORTUNITIES,
        "min_comparison_adds": MIN_COMPARISON_ADDS,
        "comparison_available": gate == "ok",
        "interpretation_gate": gate,
        "comparison_adds": n_adds,
        "buckets": buckets,
        "add_rate_underwater_with_attention": r_with,
        "add_rate_underwater_without_attention": r_without,
        "add_rate_in_profit_with_attention": rate(buckets["gain_with_attention"]),
        "add_rate_in_profit_without_attention": rate(buckets["gain_without_attention"]),
        "examples": examples,
        "interpretation": interpretation,
        "note": (f'The denominator is the opportunity-days for (instrument, trading day) pairs that are "held and in the given profit/loss state". The add-position determination uses the same ≥20% relative-change threshold as scale_events; "viewed" means that the instrument had a behavioral event within the preceding {lookback} trading days. Same-day attention is not counted (it cannot be distinguished from "decide first, view later").'),
        "caveat": ("This is a conditional frequency, not causation. More position increases on days that were viewed may also be because \"people who already intended to increase their positions check first.\" The sequence can be verified, but the motivation cannot."),
    }


def theme_lead_lag(events, weights, days):
    """How long after theme attention rises does allocation follow?

    Cross-correlates daily theme attention share against daily theme weight
    share over lags 0..14 days and reports the best-matching lag. Reported only
    when the window is long enough, since a short window can produce a strong
    correlation by chance.
    """
    if len(days) < MIN_DAYS_FOR_LAG:
        return {"available": False,
                "note": f"Observation window is only {len(days)} days, fewer than {MIN_DAYS_FOR_LAG} days; no lead-lag analysis"}
    att = defaultdict(Counter)          # date -> theme count
    for e in events:
        for th in e.get("themes") or []:
            att[e["date"]][th] += 1
    themes = {th for c in att.values() for th in c} | {ticker_info(t)["theme"] for w in weights for t in w}
    themes.discard("Other")
    out = []
    for th in themes:
        a = [att.get(d, Counter()).get(th, 0) for d in days]
        b = [_theme_share(w).get(th, 0.0) for w in weights]
        if sum(a) < 5 or max(b) <= 0:
            continue
        best = None
        for lag in range(0, 15):
            xs = a[:len(a) - lag]
            ys = b[lag:]
            if len(xs) < MIN_DAYS_FOR_LAG or sum(xs) == 0:
                continue
            ma, mb = sum(xs) / len(xs), sum(ys) / len(ys)
            num = sum((x - ma) * (y - mb) for x, y in zip(xs, ys))
            da = sum((x - ma) ** 2 for x in xs) ** 0.5
            db = sum((y - mb) ** 2 for y in ys) ** 0.5
            if not da or not db:
                continue
            r = num / (da * db)
            if best is None or r > best["correlation"]:
                best = {"lag_days": lag, "correlation": round(r, 3)}
        if best and best["correlation"] >= 0.3:
            out.append({"theme": th, "attention_events": sum(a),
                        **best, "days_used": len(days) - best["lag_days"]})
    out.sort(key=lambda x: -x["correlation"])
    return {
        "available": bool(out),
        "themes": out[:6],
        "method": "Cross-correlation between daily topic attention share and topic holding share, taking the highest correlation within 0–14 days",
        "note": ("Correlation does not equal causation and does not represent a trading decision path; it is only auxiliary evidence for the investment logic. The fewer sample days, the more likely the correlation is incidental; days_used has been labeled.") if out else
                "No themes with a correlation of 0.3 were found; no leading-lagging conclusion",
    }


def decision_funnel(events, lifecycle, entries, scale_ups, values, days):
    """attention -> research -> entry -> scale-up -> persistent holding.

    Each stage is a set of tickers with its own denominator; the stages are NOT
    a cohort tracked through a pipeline, so the rates must not be multiplied
    together. That distinction is stated in the output.
    """
    stage_by_ticker = {}
    for e in events:
        t = e.get("focus_ticker")
        if not t:
            continue
        cur = stage_by_ticker.get(t)
        if cur is None or _STAGE_RANK[e["intent_stage"]] > _STAGE_RANK[cur]:
            stage_by_ticker[t] = e["intent_stage"]

    attended = set(stage_by_ticker)
    researched = {t for t, s in stage_by_ticker.items()
                  if _STAGE_RANK[s] >= _STAGE_RANK["research"]}
    entered = {e["ticker"] for e in entries}
    scaled = {u["ticker"] for u in scale_ups}

    # persistent = the entry is still in the book 30+ days later (or at window end)
    last_index = len(days) - 1
    persistent = set()
    for en in entries:
        t, start = en["ticker"], en["index"]
        if t not in values[start]:
            continue
        still_held_at_end = t in values[last_index]
        long_enough = (last_index - start) >= 30
        if still_held_at_end and long_enough:
            persistent.add(t)

    return {
        "available": True,
        "counts": {
            "attended_tickers": len(attended),
            "reached_research": len(researched),
            "observable_entries": len(entered),
            "entries_scaled_up": len(entered & scaled),
            "entries_held_30d_plus": len(persistent),
        },
        "attention_to_research": (_safe_div(len(researched), len(attended), 3)
                                  if len(attended) >= MIN_DENOM_FOR_RATE else None),
        "research_to_entry": (_safe_div(len(entered & researched), len(researched), 3)
                              if len(researched) >= MIN_DENOM_FOR_RATE else None),
        "entry_to_scale_up": (_safe_div(len(entered & scaled), len(entered), 3)
                              if len(entered) >= MIN_ENTRIES_FOR_RATE else None),
        "entry_to_persistent_holding": (_safe_div(len(persistent), len(entered), 3)
                                        if len(entered) >= MIN_ENTRIES_FOR_RATE else None),
        "thresholds": {"min_sample": MIN_ENTRIES_FOR_RATE,
                       "min_denominator_for_rate": MIN_DENOM_FOR_RATE},
        "note": ("The denominator for each stage is independent (instruments under watch, instruments researched, observable position starts), not progressive attrition from the same batch of instruments, so the ratios cannot be multiplied together. Stages with insufficient samples return null rather than 0. Continuous holding requires the position to remain open 30 days after the position starts. Here, observable_entries includes only instruments that entered the watch sequence, so it is smaller than conversion.entries_observed (the latter includes position starts that were never watched)."),
    }


# ---------- entry point ----------

def build_attention_capital(snap, intent=None, max_examples=10):
    """Cross BM433 attention with BM453 capital. Deterministic, no LLM."""
    behaviors = snap.get("behaviors_6m") or []
    account = sorted(snap.get("account_daily") or [],
                     key=lambda d: d.get("snapshot_date") or "")
    if not behaviors or not account:
        return {"available": False,
                "note": "Requires both the BM433 behavior stream and BM453 account snapshot"}

    events = normalize_stream(behaviors)
    if not events:
        return {"available": False, "note": "BM433 event cannot be parsed"}
    build_sessions(events)

    # gate FIRST: a gap day must not be read as liquidation, and day 1 is a gap
    # day for essentially every user, so skipping this fabricates the whole
    # entry/exit history
    gated = build_days(account)
    days, weights, values, pnl, states = capital_timeline(gated)
    if not days:
        return {"available": False, "note": "BM453 has no usable snapshots"}
    gap_days = sum(1 for d in gated if d.get("carried_forward"))
    known = [d.get("book_known") for d in gated]

    baseline = baseline_index(gated)
    lifecycle = _ticker_lifecycle(days, values, baseline)
    ent, exi = entries_and_exits(days, values, lifecycle, baseline, known)
    ups, downs = scale_events(days, values, weights, known)

    attention = attention_by_ticker(events)
    # Use the last day with an actual book, not simply the final day: users who
    # flatten (or whose last day is a data gap) would otherwise be compared
    # against an empty portfolio, which makes every held ticker vanish from the
    # alignment and leaves held_at_start / dominant_state meaningless.
    latest, latest_as_of = {}, None
    for i in range(len(weights) - 1, -1, -1):
        if weights[i]:
            latest, latest_as_of = weights[i], days[i]
            break

    # If the account reports stock exposure but not one single position detail
    # was ever seen, every position-level conclusion below would be fabricated.
    # Saying nothing is the only honest option; `stock_total` alone cannot name
    # tickers, so the alignment would otherwise show a full book of 0% weights.
    max_stock = max((d.get("stock_total") or 0) for d in gated)
    if not latest and max_stock > 0:
        return {
            "available": False,
            "version": ATTENTION_CAPITAL_VERSION,
            "window": [days[0], days[-1]],
            "window_days": len(days),
            "attention_events": sum(attention.values()),
            "note": (f"During BM453, stock_total peaked at {max_stock:,.0f}, but holding details remained empty throughout, so the specific instruments could not be identified and no attention-capital cross-conclusion is output (missing data is not treated as an empty position)."),
        }
    held_now = set(latest)

    research_by_ticker = {}
    for e in events:
        t = e.get("focus_ticker")
        if not t:
            continue
        cur = research_by_ticker.get(t)
        if cur is None or _STAGE_RANK[e["intent_stage"]] > _STAGE_RANK[cur]:
            research_by_ticker[t] = e["intent_stage"]

    return {
        "available": True,
        "version": ATTENTION_CAPITAL_VERSION,
        "window_days": len(days),
        "window": [days[0], days[-1]],
        "attention_events": sum(attention.values()),
        "held_tickers_latest": len(latest),
        "positions_as_of": latest_as_of,
        "positions_as_of_note": ("Position comparison uses the last date with holdings to prevent all held instruments from disappearing when there are no positions at the end of the window."),
        "observed_entries": len(ent),
        "observed_exits": len(exi),
        "held_at_start_tickers": len([1 for v in lifecycle.values() if v["held_at_start"]]),
        "baseline_date": days[baseline] if baseline < len(days) else None,
        "baseline_note": ("The date when the first reliable position data became available; positions before this date cannot be used to determine the position entry time and are excluded from entry/exit statistics"),
        "carried_forward_days": gap_days,
        "data_quality_note": (f"{gap_days} days carried forward the previous day's holdings due to missing holdings, not counted as exit/start of a new position") if gap_days else None,
        "orientation": orientation(events, attention, held_now),
        "attention_capital_alignment": alignment(attention, latest),
        "pre_entry": pre_entry_pattern(events, ent),
        "conversion": conversion(events, lifecycle, ent, weights, days, research_by_ticker),
        "capital_conversion": capital_conversion(events, ent, weights, days),
        "persistence_pairing": persistence_pairing(events, lifecycle, days),
        "concentration": concentration_compare(attention, latest),
        "loss_conditioned_attention": loss_conditioned(events, days, values, states),
        # The companion the ratio needs: co-occurrence says nothing about whether
        # the looking preceded the acting. See attention_to_action's docstring.
        "attention_to_action": attention_to_action(events, days, states, weights,
                                                   values, known),
        "theme_lead_lag": theme_lead_lag(events, weights, days),
        "decision_funnel": decision_funnel(events, lifecycle, ent, ups, values, days),
        "position_events": {
            "entries": ent[:max_examples],
            "exits": exi[:max_examples],
            "scale_ups": ups[:max_examples],
            "scale_downs": downs[:max_examples],
        },
        "caveats": [
            "Positions already held on the first day are excluded from entry (the actual entry time cannot be determined)",
            "Attention is the proportion of behavioral events, not a psychological measure; does not constitute any investment advice",
            "The remaining analysis consists solely of behavioral descriptions; motivations must be inferred together with subsequent adds, trims, and executions",
        ],
    }


# ---------- agent-facing digest ----------

def attention_capital_digest(ac):
    """Compact view for prompts: headline numbers only, no per-ticker dumps."""
    if not ac or not ac.get("available"):
        return {"available": False,
                "note": (ac or {}).get("note", "No attention-capital cross data")}
    o = ac.get("orientation") or {}
    al = ac.get("attention_capital_alignment") or {}
    pe = ac.get("pre_entry") or {}
    cv = ac.get("conversion") or {}
    cc = ac.get("concentration") or {}
    lc = ac.get("loss_conditioned_attention") or {}
    fn = ac.get("decision_funnel") or {}
    lg = ac.get("theme_lead_lag") or {}
    pp = ac.get("persistence_pairing") or {}
    return {
        "available": True,
        "window_days": ac.get("window_days"),
        "orientation": {k: o.get(k) for k in
                        ("held_ticker_view_ratio", "nonheld_ticker_view_ratio", "interpretation")} if o else None,
        "alignment": {
            "attention_gt_capital": al.get("attention_gt_capital"),
            "capital_gt_attention": al.get("capital_gt_attention"),
            "state_counts": al.get("state_counts"),
            "largest_gaps": [{k: r[k] for k in ("ticker", "attention_share", "position_weight",
                                                "attention_capital_gap", "state")}
                             for r in (al.get("largest_gaps") or [])[:5]],
        } if al else None,
        "pre_entry": {k: pe.get(k) for k in
                      ("entries", "prior_attention_rate_7d", "median_pre_entry_attention_7d",
                       "pre_entry_research_depth_mix")} if pe.get("available") else None,
        "conversion": {k: v for k, v in cv.items() if k.endswith("conversion") or k == "universe_attended_not_held"} if cv else None,
        "concentration": cc,
        "loss_conditioned_attention": lc if lc.get("available") else None,
        "persistence_relationship_mix": pp.get("relationship_mix"),
        "theme_lead_lag": [{k: t[k] for k in ("theme", "lag_days", "correlation")}
                           for t in (lg.get("themes") or [])[:3]] if lg.get("available") else None,
        "decision_funnel": {k: fn.get(k) for k in
                            ("counts", "attention_to_research", "research_to_entry",
                             "entry_to_scale_up", "entry_to_persistent_holding")} if fn.get("available") else None,
    }
