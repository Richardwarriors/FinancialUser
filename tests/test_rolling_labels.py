"""Direct regression checks for point-in-time rolling labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rolling_labels import (build_targets, hierarchical_trade_target,
                                net_trade_action, position_actions)  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def trade(ts, side, ticker="AAA", amount=100):
    return {"executed_at": ts, "execution_state": "executed", "side": side,
            "ticker": ticker, "executed_amount": amount}


def book(holdings, stock_total=0):
    return {"status": "available", "book_known": True, "stock_total": stock_total,
            "holdings": {k: {"stock_vol": v} for k, v in holdings.items()}}


def test_et_date_and_net_action():
    print("\n[1] labels use ET date, not UTC/calendar columns")
    rows = [trade("2026-08-12T01:00:00", "buy", amount=90),  # Aug 11 21:00 ET
            trade("2026-08-12T15:00:00", "sell", amount=40)]
    out = net_trade_action(rows, "2026-08-12", observed=True)
    check("UTC rollover row excluded from Aug 12 ET", out["executed_orders"] == 1, out)
    check("remaining row labels sell", out["action"] == "sell", out)
    unknown = net_trade_action([], "2026-08-12", observed=False)
    check("unproven absence is unknown", unknown["action"] == "unknown", unknown)
    hold = net_trade_action([], "2026-08-12", observed=True)
    check("proven no execution is hold", hold["action"] == "hold", hold)


def test_multilabel_book_changes():
    print("\n[2] position targets preserve multiple ticker actions")
    before = book({"A": 10, "B": 10, "C": 10})
    after = book({"A": 15, "B": 5, "D": 4})
    out = position_actions(before, after, [], "2026-08-12",
                           before_observed=True, after_observed=True,
                           executions_observed=True)
    got = {(x["ticker"], x["action"]) for x in out["actions"]}
    expected = {("A", "add_to_existing"), ("B", "trim_position"),
                ("C", "exit_position"), ("D", "start_new_position")}
    check("all four persistent actions retained", got == expected, got)
    check("primary tie-break is deterministic", out["primary_action"] == "exit_position", out)


def test_missing_book_is_not_empty():
    print("\n[3] unknown holdings never become liquidation")
    missing = {"status": "available", "book_known": True,
               "stock_total": 500, "holdings": {}}
    out = position_actions(book({"AAA": 10}), missing, [], "2026-08-12",
                           before_observed=True, after_observed=True,
                           executions_observed=True)
    check("unknown book refuses label", out["status"] == "unknown", out)
    check("no fabricated exit", not out["actions"], out)
    out = position_actions(book({"AAA": 10}), book({"AAA": 10}), [], "2026-08-12",
                           before_observed=True, after_observed=True,
                           executions_observed=False)
    check("known unchanged books support hold", out["primary_action"] == "hold", out)


def test_round_trip_and_contract():
    print("\n[4] same-day round trips and combined contract")
    rows = [trade("2026-08-12T14:00:00", "buy", "NEW", 100),
            trade("2026-08-12T18:00:00", "sell", "NEW", 100)]
    out = position_actions(book({}), book({}), rows, "2026-08-12",
                           before_observed=True, after_observed=True,
                           executions_observed=True)
    got = [x["action"] for x in out["actions"]]
    check("flat-to-flat buy/sell is not hold",
          got == ["start_new_position", "exit_position"], got)
    both = build_targets(rows, "2026-08-12", book({}), book({}), {
        "executions_observed": True,
        "before_book_observed": True,
        "after_book_observed": True,
    })
    check("hierarchical and legacy target families emitted",
          set(both) >= {"trade_decision", "net_trade_action",
                        "position_actions", "coverage"}, both)
    check("equal amounts remain a real bidirectional trade",
          both["trade_decision"]["trade_activity"] == "trade" and
          both["trade_decision"]["trade_sides"] == "buy_and_sell" and
          both["net_trade_action"]["action"] == "hold", both)
    check("buy/sell ticker rankings are separate",
          both["trade_decision"]["buy_tickers"][0]["ticker"] == "NEW" and
          both["trade_decision"]["sell_tickers"][0]["ticker"] == "NEW", both)


def test_no_trade_and_conflict():
    print("\n[5] inactivity is observed only when execution and books agree")
    quiet = hierarchical_trade_target([], "2026-08-12",
        executions_observed=True, before_book=book({"A": 10}),
        after_book=book({"A": 10}), before_observed=True, after_observed=True)
    check("known unchanged day is no_trade", quiet["trade_activity"] == "no_trade", quiet)
    conflict = hierarchical_trade_target([], "2026-08-12",
        executions_observed=True, before_book=book({"A": 10}),
        after_book=book({"A": 5}), before_observed=True, after_observed=True)
    check("zero execution with book change is unknown",
          conflict["trade_activity"] == "unknown" and
          conflict["execution_book_consistency"] == "execution_absent_but_book_changed",
          conflict)
    ranked = hierarchical_trade_target([
        trade("2026-08-12T15:00:00", "buy", "A", 10),
        trade("2026-08-12T15:01:00", "buy", "B", 30),
        trade("2026-08-12T15:02:00", "sell", "C", 20)], "2026-08-12",
        executions_observed=True, before_book=book({}), after_book=book({}),
        before_observed=True, after_observed=True)
    check("ticker ranks use amount per side",
          [x["ticker"] for x in ranked["buy_tickers"]] == ["B", "A"] and
          [x["ticker"] for x in ranked["sell_tickers"]] == ["C"], ranked)


if __name__ == "__main__":
    test_et_date_and_net_action()
    test_multilabel_book_changes()
    test_missing_book_is_not_empty()
    test_round_trip_and_contract()
    test_no_trade_and_conflict()
    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {5 - bool(FAILS)}/5 sections")
    raise SystemExit(1 if FAILS else 0)
