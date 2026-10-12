"""Deterministic seven-trading-day user context for rolling prediction."""
from __future__ import annotations

import re
from collections import Counter
from datetime import date

from .rolling_time import last_n_trading_days, parse_event_et, parse_execution_et

_TICKER = re.compile(r"\b([A-Z]{1,6}(?:\.[A-Z])?)\b")


def _viewed(events):
    counts = Counter()
    for row in events:
        action = row.get("action_text") or ""
        if "view" not in action.lower():
            continue
        matches = _TICKER.findall(action)
        if matches:
            counts[matches[0]] += 1
    return [{"ticker": ticker, "views": count}
            for ticker, count in counts.most_common(12)]


def _action_category(action):
    """Return an English category; raw source text never enters model artifacts."""
    text = (action or "").lower()
    rules = (
        ("options_research", ("option", "期权")),
        ("news_research", ("news", "公告", "新闻")),
        ("fundamental_research", ("financial", "earning", "company info", "财报", "公司")),
        ("technical_research", ("price analysis", "chart", "technical", "技术", "走势")),
        ("quote_view", ("quote", "行情", "报价")),
        ("order_book_view", ("order book", "盘口")),
        ("navigation", ("switch to", "切换")),
    )
    for category, terms in rules:
        if any(term in text for term in terms):
            return category
    return "other"


def _known_book(row):
    holdings = row.get("holdings") or {}
    stock_total = row.get("stock_total")
    if holdings:
        return holdings
    return {} if stock_total in (None, 0, 0.0) else None


def _position_changes(rows):
    changes = Counter()
    per_ticker = []
    previous = None
    unknown = 0
    for row in rows:
        current = _known_book(row)
        if current is None:
            unknown += 1
            continue
        if previous is None:
            previous = current
            continue
        for ticker in sorted(set(previous) | set(current)):
            before = (previous.get(ticker) or {}).get("stock_vol", 0) or 0
            after = (current.get(ticker) or {}).get("stock_vol", 0) or 0
            if before == after:
                continue
            action = ("start_new_position" if before == 0 and after > 0 else
                      "exit_position" if before > 0 and after == 0 else
                      "add_to_existing" if after > before else "trim_position")
            changes[action] += 1
            per_ticker.append({"date": row.get("snapshot_date"), "ticker": ticker,
                               "action": action, "quantity_before": before,
                               "quantity_after": after})
        previous = current
    return {"counts": dict(changes), "changes": per_ticker[-40:],
            "changes_total": len(per_ticker),
            "changes_truncated": max(0, len(per_ticker) - 40),
            "unknown_book_days_skipped": unknown}


def build_recent_7d(ds, user_id, cutoff_date: date):
    days = last_n_trading_days(cutoff_date, 7)
    allowed = {d.isoformat() for d in days}
    events = [r for r in ds.get_behaviors(user_id)
              if (parse_event_et(r.get("timestamp_et")) is not None
                  and parse_event_et(r.get("timestamp_et")).date().isoformat() in allowed)]
    trades = [r for r in ds.get_trades(user_id)
              if (parse_execution_et(r.get("executed_at")) is not None
                  and parse_execution_et(r.get("executed_at")).date().isoformat() in allowed)]
    daily = [r for r in ds.get_account_daily(user_id)
             if r.get("snapshot_date") in allowed]
    actions = Counter(_action_category(r.get("action_text")) for r in events)
    executed = [r for r in trades if r.get("execution_state") == "executed"]
    sides = Counter(r.get("side") or "unknown" for r in executed)
    tickers = Counter(r.get("ticker") for r in executed if r.get("ticker"))
    trade_dates = {parse_execution_et(r.get("executed_at")).date().isoformat()
                   for r in executed if parse_execution_et(r.get("executed_at"))}
    app_dates = {parse_event_et(r.get("timestamp_et")).date().isoformat()
                 for r in events if parse_event_et(r.get("timestamp_et"))}
    no_trade_days = [d for d in days if d.isoformat() not in trade_dates]
    streak = 0
    for day in reversed(days):
        if day.isoformat() in trade_dates:
            break
        streak += 1
    side_days = Counter()
    for day in days:
        day_sides = {r.get("side") for r in executed
                     if parse_execution_et(r.get("executed_at")) and
                     parse_execution_et(r.get("executed_at")).date() == day}
        if day_sides == {"buy", "sell"}:
            side_days["buy_and_sell"] += 1
        elif day_sides == {"buy"}:
            side_days["buy_only"] += 1
        elif day_sides == {"sell"}:
            side_days["sell_only"] += 1
    return {
        "trading_days": [d.isoformat() for d in days],
        "window_start": days[0].isoformat(), "window_end": days[-1].isoformat(),
        "behavior": {
            "events": len(events),
            "active_days": len({parse_event_et(r.get("timestamp_et")).date()
                                for r in events}),
            "top_actions": [{"action": k, "count": v}
                            for k, v in actions.most_common(12)],
            "viewed_tickers": _viewed(events),
        },
        "trade_activity": {
            "trading_days_observed": len(days),
            "days_with_executions": len(trade_dates),
            "days_without_executions": len(no_trade_days),
            "execution_day_rate": round(len(trade_dates) / len(days), 4),
            "current_no_trade_streak": streak,
            "app_active_without_execution_days": len(app_dates - trade_dates),
            "trade_side_days": dict(side_days),
            "note": ("historical cutoff-side summaries only; a zero day is descriptive "
                     "and does not prove target-day inactivity"),
        },
        "ticker_candidates": {
            "source": "cutoff-side recent executions and app views",
            "recently_traded": [k for k, _ in tickers.most_common(20)],
            "recently_viewed": [x["ticker"] for x in _viewed(events)],
        },
        "trades": {
            "rows": len(trades), "executed_by_side": dict(sides),
            "top_tickers": [{"ticker": k, "orders": v}
                            for k, v in tickers.most_common(10)],
            "orders": [{"executed_at_et": r.get("executed_at_et"),
                        "ticker": r.get("ticker"), "side": r.get("side"),
                        "quantity": r.get("executed_quantity"),
                        "amount": r.get("executed_amount"),
                        "state": r.get("execution_state")} for r in trades[-50:]],
            "orders_total": len(trades), "orders_truncated": max(0, len(trades) - 50),
        },
        "portfolio_changes": _position_changes(daily),
        "coverage": {"behavior_rows": len(events), "trade_rows": len(trades),
                     "snapshot_days": len(daily), "expected_trading_days": 7},
    }
