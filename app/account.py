"""Account analytics: external cash flow detection and flow-adjusted returns.

Why this module exists
----------------------
account_snapshots's `asset_total` moves for two very different reasons: the market moved, or
the user paid money in / took money out. Measuring return or drawdown on the raw
series therefore conflates investment performance with savings behaviour.

Returns and drawdowns are always computed on a **time-weighted** curve with
the flows removed, and the raw numbers are kept alongside for audit.

Flow detection rule
-------------------
Only days with no trade AND an unchanged holding book are considered. On such a
day a buy/sell cannot have happened, so price movement hits `asset_total` and
`stock_total` identically and:

    external_flow ≈ Δasset_total − Δstock_total

Any internal transfer (buy/sell) is excluded by the no-trade filter, so a
portfolio rebalance can never be mistaken for a deposit. The estimate is cross
-checked against Δcash and discarded when the two disagree, because a
disagreement means the day was not as clean as it looked.
"""
from .taxonomy import parse_ts

_TS = {}

# A curve built from less than half the days cannot support a return or a
# drawdown claim. Without this guard an account that opened at 0 and was funded
# later reports "drawdown 0%" — false precision, and worse than saying unknown.
MIN_COVERAGE = 0.5


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _trade_days(trades):
    """Days on which a trade executed.

    Uses `executed_at` (recovered from the payload). The `trade_date` column is
    the label extraction date and would mark the whole batch as a single
    trading day, which would in turn hide nearly every real flow day.
    """
    out = set()
    for t in trades or []:
        d = str(t.get("executed_at") or t.get("trade_date") or "")[:10]
        if d:
            out.add(d)
    return out


def external_flows(acct, trades, min_amount=1.0, cross_check_tol=0.2):
    """Detect external cash flows. Returns a list of {date, amount, direction}.

    Positive amount = money in. Ordered by date.
    """
    if not acct or len(acct) < 3:
        return []
    trade_days = _trade_days(trades)
    flows = []
    for prev, cur in zip(acct, acct[1:]):
        d = cur.get("snapshot_date")
        if not d or d in trade_days:
            continue
        q_prev = {k: _f(v.get("stock_vol"), 0.0) for k, v in (prev.get("holdings") or {}).items()}
        q_cur = {k: _f(v.get("stock_vol"), 0.0) for k, v in (cur.get("holdings") or {}).items()}
        if set(q_prev) != set(q_cur) or any(abs(q_prev[k] - q_cur.get(k, 0.0)) > 1e-9 for k in q_prev):
            continue
        dn = (_f(cur.get("asset_total"), 0.0) or 0.0) - (_f(prev.get("asset_total"), 0.0) or 0.0)
        ds = (_f(cur.get("stock_total"), 0.0) or 0.0) - (_f(prev.get("stock_total"), 0.0) or 0.0)
        dcash = (_f(cur.get("money_remain"), 0.0) or 0.0) - (_f(prev.get("money_remain"), 0.0) or 0.0)
        flow = dn - ds
        if abs(flow) < min_amount:
            continue
        # the cash line must move with the estimated flow, or the day is not clean
        if abs(flow - dcash) > max(min_amount, abs(dcash) * cross_check_tol):
            continue
        flows.append({"date": d, "amount": round(flow, 2),
                      "direction": "in" if flow > 0 else "out"})
    return flows


def adjusted_equity_series(acct, trades):
    """Time-weighted equity curve with external flows neutralised.

    Each day's return is `(V_t − flow_t) / V_{t-1} − 1`, chained forward. A
    deposit therefore leaves the index unchanged instead of manufacturing a
    gain.

    Degenerate steps are skipped rather than propagated: when the previous
    balance is too small to divide by (a brand-new or drained account), the
    ratio is meaningless and would otherwise produce the astronomic numbers
    seen in the raw series. Skipped steps are counted and reported.
    """
    if not acct:
        return {"dates": [], "values": [], "flows": [], "flow_net": 0.0,
                "flow_count": 0, "skipped_steps": 0}
    flows = external_flows(acct, trades)
    flow_by_date = {f["date"]: f["amount"] for f in flows}

    vals = [_f(d.get("asset_total")) for d in acct]
    positives = [v for v in vals if v and v > 0]
    if not positives:
        return {"dates": [d.get("snapshot_date") for d in acct], "values": vals,
                "flows": flows, "flow_net": sum(f["amount"] for f in flows),
                "flow_count": len(flows), "skipped_steps": 0}
    # a base below 0.1% of the largest balance cannot anchor a meaningful return
    min_base = max(1.0, 0.001 * max(positives))

    dates, out = [], []
    idx = None
    skipped = 0
    for i, d in enumerate(acct):
        date = d.get("snapshot_date")
        v = vals[i]
        if v is None:
            continue
        if idx is None:
            idx = v
            dates.append(date)
            out.append(v)
            continue
        prev = vals[i - 1] if i > 0 else None
        if prev is None or prev < min_base:
            skipped += 1
            continue
        flow = flow_by_date.get(date, 0.0)
        r = (v - flow) / prev - 1
        # a single-day move beyond ±50% after removing flows is a data artifact,
        # not an investment result
        if abs(r) > 0.5:
            skipped += 1
            continue
        idx = idx * (1 + r)
        dates.append(date)
        out.append(round(idx, 4))

    return {"dates": dates, "values": out, "flows": flows,
            "flow_net": round(sum(f["amount"] for f in flows), 2),
            "flow_count": len(flows), "skipped_steps": skipped,
            "min_base": round(min_base, 2),
            "coverage": round(len(out) / len(acct), 3) if acct else 0.0,
            "reliable": bool(acct) and len(out) >= 2 and len(out) / len(acct) >= MIN_COVERAGE}


def max_drawdown(dates, values):
    """(drawdown, date, peak_value, recovered_by_date) for a curve."""
    if not values:
        return 0.0, None, None, None
    peak, peak_val = values[0], values[0]
    worst, worst_date, peak_at = 0.0, None, values[0]
    recover = None
    for d, v in zip(dates, values):
        if v > peak:
            if worst_date and recover is None and worst < -0.02:
                recover = d
            peak, peak_val = v, v
        if peak:
            dd = v / peak - 1
            if dd < worst:
                worst, worst_date, peak_at = dd, d, peak
    return round(worst, 4), worst_date, round(peak_at, 4), recover


def total_return(values):
    if not values or not values[0]:
        return None
    return round(values[-1] / values[0] - 1, 4)


def daily_returns(values):
    return [(b / a - 1) for a, b in zip(values, values[1:]) if a]
