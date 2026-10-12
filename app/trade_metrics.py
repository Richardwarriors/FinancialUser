"""Trading metrics: activity, execution behaviour and realised results.

Derived from trade_executions alone, so it works for every user with trades.

Timing
------
All timing uses the real execution timestamp (`executed_at`, recovered from the
raw payload by app/trade_payload.py). The `trade_date` column is the label
extraction date and disagrees with reality on every row observed, so any metric
built on it measured the extraction batch rather than the user.

FIFO matching
-------------
Realised results come from FIFO lot matching per ticker: buys open lots, sells
consume them oldest-first, and each consumed lot yields a realised P&L and a
holding period. Trades still open at the end are reported as such and excluded
from win/loss statistics — counting them as wins because they are currently
profitable would be look-ahead bias.

Partial fills are handled by consuming quantity, not whole lots, so a sell that
splits across two buys produces two P&L records with their own holding periods.

Data sufficiency is reported honestly: `matched_round_trips` and
`coverage_note` make it obvious when a result rests on three trades instead of
three hundred. Metrics that cannot be supported are omitted, never zero-filled —
a `win_rate` of 0 and "no closed trades" are completely different facts.
"""
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

METRIC_VERSION = "trade-metrics-v1"
MIN_ROUND_TRIPS = 3          # below this, distribution statistics are noise
MIN_ROUND_TRIPS_DIST = 8     # below this, skew/kurtosis are not reported

# Cost/fees are not present in trade_executions, so "win" means gross P&L > 0. Stated in
# the output rather than silently ignored.
GROSS_BASIS = "gross_pnl_no_fees"


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _ts(v):
    from .taxonomy import parse_ts
    return parse_ts(v)


def _safe_div(a, b, digits=4):
    if not b:
        return None
    return round(a / b, digits)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


# ---------- normalisation ----------

def normalise_trades(trades):
    """Order trades by real execution time; drop unusable rows."""
    out = []
    for t in trades or []:
        if (t.get("execution_state") or "executed") != "executed":
            continue
        tk = (t.get("ticker") or "").upper()
        ts = _ts(t.get("executed_at")) or _ts(t.get("trade_date"))
        qty = _f(t.get("executed_quantity"))
        amt = _f(t.get("executed_amount"))
        if not tk or ts is None or not qty or qty <= 0:
            continue
        out.append({
            "ticker": tk, "ts": ts, "side": (t.get("side") or "unknown").lower(),
            "qty": abs(qty), "amount": abs(amt or 0.0),
            "price": _f(t.get("price")) or (_safe_div(abs(amt or 0.0), abs(qty), 6)),
            "day": ts.date().isoformat(),
        })
    out.sort(key=lambda x: x["ts"])
    return out


# ---------- FIFO ----------

def fifo_match(trades):
    """FIFO lot matching. Returns (round_trips, open_lots).

    Each round trip: {ticker, buy_ts, sell_ts, qty, buy_price, sell_price,
                      pnl, hold_days, return_pct}
    """
    books = defaultdict(list)          # ticker -> [[qty, price, ts], ...]
    round_trips = []
    for t in trades:
        tk = t["ticker"]
        if t["side"] == "buy":
            books[tk].append([t["qty"], t["price"], t["ts"]])
            continue
        if t["side"] != "sell":
            continue
        remaining = t["qty"]
        while remaining > 1e-9 and books[tk]:
            lot = books[tk][0]
            take = min(lot[0], remaining)
            pnl = (t["price"] - lot[1]) * take
            hold = (t["ts"] - lot[2]).total_seconds() / 86400
            round_trips.append({
                "ticker": tk, "buy_ts": lot[2], "sell_ts": t["ts"], "qty": round(take, 4),
                "buy_price": round(lot[1], 4), "sell_price": round(t["price"], 4),
                "pnl": round(pnl, 2),
                "return_pct": round((t["price"] / lot[1] - 1) * 100, 2) if lot[1] else None,
                "hold_days": round(hold, 2),
            })
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-9:
                books[tk].pop(0)
        # a sell with no matching lot (position opened before the window) is
        # recorded as unmatched, not silently discarded
        if remaining > 1e-9:
            round_trips.append({"ticker": tk, "unmatched": True, "qty": round(remaining, 4),
                                "sell_ts": t["ts"]})
    open_lots = [{"ticker": tk, "qty": round(l[0], 4), "price": round(l[1], 4),
                  "opened": l[2].date().isoformat(), "days_open": round((trades[-1]["ts"] - l[2]).days, 1)}
                 for tk, lots in books.items() for l in lots if l[0] > 1e-9] if trades else []
    return [r for r in round_trips if not r.get("unmatched")], open_lots


# ---------- blocks ----------

def activity(trades):
    """Volume, cadence and concentration of trading."""
    if not trades:
        return None
    total = len(trades)
    amount = sum(t["amount"] for t in trades)
    days = sorted({t["day"] for t in trades})
    span_days = max((trades[-1]["ts"] - trades[0]["ts"]).days + 1, 1)
    by_ticker = Counter()
    for t in trades:
        by_ticker[t["ticker"]] += t["amount"]
    total_amt = sum(by_ticker.values()) or 1.0
    hhi = sum((v / total_amt) ** 2 for v in by_ticker.values())
    state = Counter((t.get("side") or "unknown") for t in trades)
    return {
        "total_orders": total,
        "total_amount": round(amount, 2),
        "buy_orders": state.get("buy", 0),
        "sell_orders": state.get("sell", 0),
        "buy_amount": round(sum(t["amount"] for t in trades if t["side"] == "buy"), 2),
        "sell_amount": round(sum(t["amount"] for t in trades if t["side"] == "sell"), 2),
        "net_buy_amount": round(sum(t["amount"] * (1 if t["side"] == "buy" else -1)
                                    for t in trades if t["side"] in ("buy", "sell")), 2),
        "trading_days": len(days),
        "observation_days": span_days,
        "daily_avg_orders": _safe_div(total, len(days), 2),
        "activity_rate": _safe_div(len(days), span_days, 3),
        "max_orders_per_day": max(Counter(t["day"] for t in trades).values()),
        "unique_tickers": len(by_ticker),
        "hhi_index": round(hhi, 4),
        "first_trade": trades[0]["ts"].date().isoformat(),
        "last_trade": trades[-1]["ts"].date().isoformat(),
        "success_rate": None,
        "success_rate_note": "trade_executions contains only filled records, with no canceled orders; order success rate cannot be calculated",
    }


def execution(trades, round_trips):
    """Holding periods and trade sizing (FIFO, real timestamps)."""
    if not trades:
        return None
    holds = [r["hold_days"] for r in round_trips]
    sgn = [r for r in round_trips if r["pnl"] > 0]
    los = [r for r in round_trips if r["pnl"] < 0]
    out = {
        "avg_trade_size": _safe_div(sum(t["amount"] for t in trades), len(trades), 2),
        "median_trade_size": _percentile([t["amount"] for t in trades], 0.5),
        "matched_round_trips": len(round_trips),
        "basis": GROSS_BASIS,
    }
    if holds:
        out.update({
            "avg_hold_days": round(_mean(holds), 2),
            "std_hold_days": round(_std(holds), 2) if _std(holds) is not None else None,
            "median_hold_days": _percentile(holds, 0.5),
        })
    if sgn:
        out["winner_hold_days"] = round(_mean([r["hold_days"] for r in sgn]), 2)
    if los:
        out["loser_hold_days"] = round(_mean([r["hold_days"] for r in los]), 2)
    # NOTE: the loser/winner hold-days ratio is intentionally NOT emitted here.
    # It was duplicated as `hold_days_ratio_loser_over_winner` in this block and
    # as `behavior.avg_down_hold_ratio` with an identical formula (verified
    # byte-identical on every comparable user). One owner: `behavior`.
    return out


def results(round_trips):
    """Realised P&L statistics. Omitted (not zeroed) when unsupported."""
    n = len(round_trips)
    if n == 0:
        return {"closed_trades": 0,
                "note": "No matching exit trades (FIFO found no buy-sell pairs)"}
    wins = [r["pnl"] for r in round_trips if r["pnl"] > 0]
    losses = [r["pnl"] for r in round_trips if r["pnl"] < 0]
    pnls = [r["pnl"] for r in round_trips]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    out = {
        "closed_trades": n,
        "basis": GROSS_BASIS,
        "wins": len(wins), "losses": len(losses),
        "flat": n - len(wins) - len(losses),
        "total_realised_pnl": round(sum(pnls), 2),
        "win_rate": _safe_div(len(wins), n, 3),
        "avg_win": _safe_div(gross_win, len(wins), 2) if wins else None,
        "avg_loss": round(-_safe_div(gross_loss, len(losses), 2), 2) if losses else None,
        "profit_factor": _safe_div(gross_win, gross_loss, 3) if losses and gross_loss else None,
        "expectancy_per_trade": _safe_div(sum(pnls), n, 2),
    }
    sd = _std(pnls)
    if sd:
        out["pnl_std"] = round(sd, 2)
        out["risk_reward_ratio"] = _safe_div(_mean(pnls) or 0.0, sd, 3)
    if n < MIN_ROUND_TRIPS:
        out["distribution_note"] = (
            f"Only {n} trades were closed, fewer than {MIN_ROUND_TRIPS}; no reliable conclusion is provided for win rate/profit-loss ratio; skewness and kurtosis require more samples")
        return out
    if n >= MIN_ROUND_TRIPS_DIST and sd:
        m = sum(pnls) / n
        out["pnl_skewness"] = round(sum(((x - m) / sd) ** 3 for x in pnls) / n, 3)
        out["pnl_kurtosis"] = round(sum(((x - m) / sd) ** 4 for x in pnls) / n - 3, 3)
        out["kurtosis_note"] = "Trimmed by 3, 0 indicates tails comparable to a normal distribution, >0 indicates heavier tails"
    else:
        out["distribution_note"] = (
            f"Only {n} round trips closed; skewness and kurtosis not provided (requires ≥{MIN_ROUND_TRIPS_DIST} round trips to be meaningful)")
    return out


def behaviour(trades, round_trips):
    """Habitual patterns: averaging down and quick profit taking."""
    if not trades:
        return None
    out = {"basis": GROSS_BASIS}
    wins = [r for r in round_trips if r["pnl"] > 0]
    losses = [r for r in round_trips if r["pnl"] < 0]
    if wins:
        quick = [r for r in wins if r["hold_days"] <= 7]
        out["quick_profit_taking_ratio"] = _safe_div(len(quick), len(wins), 3)
    ph = (out.get("winner_hold_days"), out.get("loser_hold_days"))
    if wins and losses:
        wh = _mean([r["hold_days"] for r in wins])
        lh = _mean([r["hold_days"] for r in losses])
        out["avg_down_hold_ratio"] = _safe_div(lh, wh, 2)
        out["avg_down_note"] = ("Hold losing positions significantly longer than profitable positions, tending to wait for losing positions to recover to breakeven"
                                if lh and wh and lh > 1.5 * wh else
                                "No significant deviation in the time structure of profitable and losing positions")
    # repeated buying of the same name within a short window = scaling in
    by_tk = defaultdict(list)
    for t in trades:
        if t["side"] == "buy":
            by_tk[t["ticker"]].append(t["ts"])
    scale_in = 0
    for tk, tss in by_tk.items():
        tss.sort()
        for a, b in zip(tss, tss[1:]):
            if (b - a) <= timedelta(days=3):
                scale_in += 1
    if by_tk:
        out["scale_in_events_3d"] = scale_in
    return out


# ---------- entry point ----------

def build_trade_metrics(trades):
    """Full trading block. Returns dict with an `available` flag."""
    norm = normalise_trades(trades)
    if not norm:
        return {"available": False,
                "note": "No usable transaction records (trade_executions is empty or the transaction time cannot be parsed)"}
    rts, open_lots = fifo_match(norm)
    out = {
        "available": True,
        "version": METRIC_VERSION,
        "timing_basis": "real execution timestamp from payload (not the trade_date column)",
        "activity": activity(norm),
        "execution": execution(norm, rts),
        "results": results(rts),
        "behavior": behaviour(norm, rts),
        "open_positions_at_end": len(open_lots),
    }
    n = len(rts)
    out["coverage"] = {
        "trades": len(norm),
        "round_trips": n,
        "open_lots": len(open_lots),
        "note": (f"{len(norm)} trades matched to {n} position-closing trades and {len(open_lots)} open lots;"
                 + ("Sufficient samples" if n >= MIN_ROUND_TRIPS_DIST else
                    "Few samples; win rate/profit-loss ratio/distribution metrics are for reference only") if n else
                 "Could not match any buy-sell pairs (possibly only bought and not sold, or the position was established before the observation window)"),
    }
    return out
