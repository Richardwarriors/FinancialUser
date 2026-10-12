"""Point-in-time labels for next-trading-day rolling prediction.

The functions in this module are pure: callers supply the target day's executions,
the books immediately before and after that day, and explicit source-coverage flags.
Absence is never treated as an observed zero. In particular, an unknown/missing
holdings payload cannot become a fabricated liquidation or a negative ``hold`` label.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
POSITION_ACTIONS = (
    "start_new_position",
    "add_to_existing",
    "trim_position",
    "exit_position",
    "hold",
)
# Stable tie-break only. The primary label is otherwise selected by number of tickers.
_PRIMARY_PRIORITY = {
    "exit_position": 0,
    "start_new_position": 1,
    "trim_position": 2,
    "add_to_existing": 3,
    "hold": 4,
}


def _date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def execution_time_et(row: dict) -> datetime | None:
    """Return an execution timestamp in America/New_York.

    BM452 ``executed_at`` is naive UTC in this corpus. An offset-aware value is
    honoured as written and converted to ET. Invalid or absent timestamps are not
    replaced with a date-only field because that would silently change day labels.
    """
    value = row.get("executed_at")
    if not value:
        return None
    try:
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(NEW_YORK)


def executions_on_target(executions: list[dict], target_date) -> tuple[list[dict], int]:
    """Select executed buy/sell rows by their true ET trading date."""
    target = _date(target_date)
    rows, invalid = [], 0
    for row in executions or []:
        if row.get("execution_state", "executed") != "executed":
            continue
        if row.get("side") not in ("buy", "sell"):
            continue
        et = execution_time_et(row)
        if et is None:
            invalid += 1
        elif et.date() == target:
            rows.append(row)
    return rows, invalid


def net_trade_action(executions: list[dict], target_date, *, observed: bool) -> dict:
    """Build the target day's net buy/sell/hold label from executed amount."""
    target = _date(target_date).isoformat()
    if not observed:
        return {
            "status": "unknown",
            "action": "unknown",
            "is_observed": False,
            "target_date": target,
            "reason": "target-day execution coverage is not proven",
        }

    rows, invalid = executions_on_target(executions, target_date)
    buy = sum(abs(r.get("executed_amount") or r.get("executed_amount_real") or 0)
              for r in rows if r.get("side") == "buy")
    sell = sum(abs(r.get("executed_amount") or r.get("executed_amount_real") or 0)
               for r in rows if r.get("side") == "sell")
    action = "buy" if buy > sell else "sell" if sell > buy else "hold"
    by_ticker = defaultdict(lambda: {"buy_amount": 0.0, "sell_amount": 0.0,
                                     "buy_orders": 0, "sell_orders": 0})
    for row in rows:
        ticker = row.get("ticker") or "__unknown__"
        side = row["side"]
        amount = abs(row.get("executed_amount") or row.get("executed_amount_real") or 0)
        by_ticker[ticker][f"{side}_amount"] += amount
        by_ticker[ticker][f"{side}_orders"] += 1

    return {
        "status": "observed",
        "action": action,
        "is_observed": True,
        "target_date": target,
        "buy_amount": round(buy, 2),
        "sell_amount": round(sell, 2),
        "executed_orders": len(rows),
        "invalid_timestamp_rows_ignored": invalid,
        "tickers": [{"ticker": ticker, **values}
                    for ticker, values in sorted(by_ticker.items())],
        "note": ("hold means proven execution coverage with no net amount; it is not "
                 "used when target-day coverage is unknown"),
    }


def _known_book(book: dict | None) -> dict[str, float] | None:
    """Return ticker quantities only for an explicitly known book.

    An empty holdings mapping is valid only when the caller explicitly marks the book
    available/known. ``stock_total > 0`` with empty holdings is rejected as unknown.
    """
    if not isinstance(book, dict):
        return None
    known = book.get("book_known") is True or book.get("status") in ("available", "known")
    if not known:
        return None
    holdings = book.get("holdings")
    if not isinstance(holdings, dict):
        return None
    if not holdings and (book.get("stock_total") or 0) > 0:
        return None
    out = {}
    for ticker, raw in holdings.items():
        if isinstance(raw, dict):
            qty = raw.get("stock_vol")
            if qty is None:
                qty = raw.get("quantity")
        else:
            qty = raw
        if isinstance(qty, (int, float)) and qty > 0:
            out[str(ticker)] = float(qty)
    return out


def _ticker_sides(rows: list[dict]) -> dict[str, set[str]]:
    out = defaultdict(set)
    for row in rows:
        if row.get("ticker") and row.get("side") in ("buy", "sell"):
            out[str(row["ticker"])].add(row["side"])
    return out


def position_actions(before_book: dict | None, after_book: dict | None,
                     executions: list[dict], target_date, *,
                     before_observed: bool, after_observed: bool,
                     executions_observed: bool) -> dict:
    """Build multi-label position actions from known books plus intraday executions.

    End-of-day book deltas are authoritative for persistent changes. If both books
    are unchanged but proven target-day executions contain both sides, a same-day
    round trip is represented by two actions rather than being collapsed to ``hold``.
    """
    target = _date(target_date).isoformat()
    before = _known_book(before_book) if before_observed else None
    after = _known_book(after_book) if after_observed else None
    if before is None or after is None:
        return {
            "status": "unknown",
            "is_observed": False,
            "target_date": target,
            "actions": [],
            "primary_action": "unknown",
            "reason": "both boundary books must be explicitly observed and known",
            "limitations": [
                "missing or structurally empty holdings are not interpreted as an empty book",
                "daily books cannot reveal intraday position paths",
            ],
        }

    rows, invalid = executions_on_target(executions, target_date) \
        if executions_observed else ([], 0)
    sides = _ticker_sides(rows)
    actions = []
    for ticker in sorted(set(before) | set(after)):
        b, a = before.get(ticker, 0.0), after.get(ticker, 0.0)
        action = None
        if b == 0 and a > 0:
            action = "start_new_position"
        elif b > 0 and a == 0:
            action = "exit_position"
        elif a > b:
            action = "add_to_existing"
        elif 0 < a < b:
            action = "trim_position"
        if action:
            actions.append({"ticker": ticker, "action": action,
                            "before_quantity": b, "after_quantity": a,
                            "source": "boundary_book_delta"})

    # A round trip leaves no EOD delta. Preserve what can be established from the
    # execution stream without pretending that its exact intraday path is known.
    if executions_observed:
        changed = {a["ticker"] for a in actions}
        for ticker in sorted(sides):
            if ticker in changed or sides[ticker] != {"buy", "sell"}:
                continue
            qty = before.get(ticker, 0.0)
            pair = (("start_new_position", "exit_position") if qty == 0 else
                    ("add_to_existing", "trim_position"))
            for action in pair:
                actions.append({"ticker": ticker, "action": action,
                                "before_quantity": qty,
                                "after_quantity": after.get(ticker, 0.0),
                                "source": "executed_same_day_round_trip"})

    if not actions:
        actions = [{"ticker": None, "action": "hold",
                    "source": "observed_unchanged_boundary_books"}]

    counts = Counter(a["action"] for a in actions)
    primary = sorted(counts, key=lambda a: (-counts[a], _PRIMARY_PRIORITY[a]))[0]
    return {
        "status": "observed",
        "is_observed": True,
        "target_date": target,
        "actions": actions,
        "primary_action": primary,
        "action_counts": dict(counts),
        "invalid_timestamp_rows_ignored": invalid,
        "limitations": [
            "boundary books observe persistent position changes, not the exact intraday path",
            "same-day round trips are inferred only when both executed sides are observed",
            "corporate actions and transfers require separate adjustment metadata",
        ],
    }


def _book_changed(before_book: dict | None, after_book: dict | None,
                  before_observed: bool, after_observed: bool) -> bool | None:
    """Whether known boundary quantities changed; None means it cannot be known."""
    before = _known_book(before_book) if before_observed else None
    after = _known_book(after_book) if after_observed else None
    if before is None or after is None:
        return None
    return before != after


def hierarchical_trade_target(executions: list[dict], target_date, *,
                              executions_observed: bool,
                              before_book: dict | None, after_book: dict | None,
                              before_observed: bool, after_observed: bool) -> dict:
    """Activity first, then observed sides and amount-ranked ticker targets.

    A zero-row execution day is only a genuine ``no_trade`` when the two known
    boundary books are unchanged. Zero executions plus a changed (or unavailable)
    book is deliberately unknown: incomplete execution capture must never train the
    model to predict inactivity.
    """
    target = _date(target_date).isoformat()
    if not executions_observed:
        return {"status": "unknown", "is_observed": False,
                "trade_activity": "unknown", "trade_sides": "unknown",
                "buy_tickers": [], "sell_tickers": [], "target_date": target,
                "execution_book_consistency": "unknown",
                "reason": "target-day execution coverage is not proven"}
    rows, invalid = executions_on_target(executions, target_date)
    changed = _book_changed(before_book, after_book, before_observed, after_observed)
    if not rows:
        if changed is not False:
            return {"status": "unknown", "is_observed": False,
                    "trade_activity": "unknown", "trade_sides": "unknown",
                    "buy_tickers": [], "sell_tickers": [], "target_date": target,
                    "execution_book_consistency": ("execution_absent_but_book_changed"
                                                   if changed else "book_unknown"),
                    "invalid_timestamp_rows_ignored": invalid,
                    "reason": "zero executions do not agree with a known unchanged book"}
        return {"status": "observed", "is_observed": True,
                "trade_activity": "no_trade", "trade_sides": "not_applicable",
                "buy_tickers": [], "sell_tickers": [], "target_date": target,
                "execution_book_consistency": "consistent",
                "executed_orders": 0,
                "invalid_timestamp_rows_ignored": invalid}

    by_side = {"buy": defaultdict(lambda: {"amount": 0.0, "orders": 0}),
               "sell": defaultdict(lambda: {"amount": 0.0, "orders": 0})}
    for row in rows:
        ticker = str(row.get("ticker") or "__unknown__")
        side = row["side"]
        amount = abs(row.get("executed_amount") or
                     row.get("executed_amount_real") or 0)
        by_side[side][ticker]["amount"] += amount
        by_side[side][ticker]["orders"] += 1

    def ranking(side):
        return [{"ticker": ticker, "amount": round(values["amount"], 2),
                 "orders": values["orders"]}
                for ticker, values in sorted(by_side[side].items(),
                    key=lambda item: (-item[1]["amount"], -item[1]["orders"], item[0]))]

    buys, sells = ranking("buy"), ranking("sell")
    sides = ("buy_and_sell" if buys and sells else
             "buy_only" if buys else "sell_only")
    consistency = ("consistent" if changed is not False else
                   "executions_present_but_book_unchanged")
    return {"status": "observed", "is_observed": True,
            "trade_activity": "trade", "trade_sides": sides,
            "buy_tickers": buys, "sell_tickers": sells,
            "target_date": target, "executed_orders": len(rows),
            "execution_book_consistency": consistency,
            "invalid_timestamp_rows_ignored": invalid}


def build_targets(executions: list[dict], target_date, before_book: dict | None,
                  after_book: dict | None, coverage: dict) -> dict:
    """Build hierarchical activity/ticker targets plus legacy audit families."""
    trades_ok = coverage.get("executions_observed") is True
    before_ok = coverage.get("before_book_observed") is True
    after_ok = coverage.get("after_book_observed") is True
    return {
        "trade_decision": hierarchical_trade_target(
            executions, target_date, executions_observed=trades_ok,
            before_book=before_book, after_book=after_book,
            before_observed=before_ok, after_observed=after_ok),
        # Kept for backwards audit comparisons, not the primary v7 task.
        "net_trade_action": net_trade_action(
            executions, target_date, observed=trades_ok),
        "position_actions": position_actions(
            before_book, after_book, executions, target_date,
            before_observed=before_ok, after_observed=after_ok,
            executions_observed=trades_ok),
        "coverage": dict(coverage),
        "timezone": "America/New_York",
    }
