#!/usr/bin/env python3
"""Regression tests for the point-in-time rolling market block."""
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rolling_market import (MOVERS_K, WATCHLIST_K, bars_at_cutoff,
                                build_market_context, market_features,
                                underlying_symbol)


def series(symbol, start=date(2026, 7, 27), n=14, base=100):
    rows = []
    cur = start
    index = 0
    while len(rows) < n:
        if cur.weekday() < 5:
            close = base + index
            rows.append({"date": cur.isoformat(), "close": close,
                         "high": close + 1, "low": close - 1,
                         "volume": 1000 + 100 * index})
            index += 1
        cur += timedelta(days=1)
    return rows


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print("PASS", name)


def main():
    cutoff = date(2026, 8, 7)
    rows = series("SPY")
    selected = bars_at_cutoff(rows, cutoff)
    check("exact seven bars", len(selected) == 7, selected)
    check("never reads target/future bar",
          all(row["date"] <= cutoff.isoformat() for row in selected), selected[-1])
    feature = market_features(rows, cutoff)
    check("feature boundary", feature["as_of_date"] == cutoff.isoformat(), feature)
    check("window is seven trading sessions", feature["bars_used"] == 7, feature)

    option = "NVDA20260807C225"
    check("option maps to underlying",
          underlying_symbol(option) == ("NVDA", "underlying"),
          underlying_symbol(option))

    cached = {
        "SPY": series("SPY", base=100),
        "QQQ": series("QQQ", base=200),
        "AAPL": series("AAPL", base=50),
        "NVDA": series("NVDA", base=150),
        "MSFT": series("MSFT", base=80),
        "TSLA": series("TSLA", base=70),
    }
    positions = [
        {"ticker": "AAPL", "market_value": 600},
        {"ticker": option, "market_value": 300},
        {"ticker": "MISSING", "market_value": 100},
        {"ticker": "MSFT", "market_value": 200},
        {"ticker": "TSLA", "market_value": 100},
    ]
    watchlist = [{"ticker": ticker, "views": 20 - i}
                 for i, ticker in enumerate(("TSLA", "AMD", "GOOG", "META", "AMZN",
                                             "NFLX", "INTC", "ORCL"))]
    context = build_market_context(cutoff, positions, watchlist,
                                   lambda symbol: cached.get(symbol, []))
    check("point in time audit", context["point_in_time"]["clean"], context)
    check("two benchmark regime", set(context["regime"]) == {"SPY", "QQQ"},
          context["regime"])
    check("portfolio is weighted aggregate",
          context["portfolio"]["status"] == "available" and
          "weighted_return_window" in context["portfolio"], context["portfolio"])
    check("missing price is explicit",
          "MISSING" in context["portfolio"]["missing_prices"], context["portfolio"])
    check("movers capped symmetrically",
          len(context["movers"]["best"]) <= MOVERS_K and
          len(context["movers"]["worst"]) <= MOVERS_K, context["movers"])
    check("watchlist cap", len(context["watchlist"]["top"]) <= WATCHLIST_K,
          context["watchlist"])
    check("option proxy disclosed",
          {"source": option, "priced_as": "NVDA", "proxy_type": "underlying"}
          in context["coverage"]["proxy_mappings"], context["coverage"])
    check("budgets reported",
          all("chars" in value and "limit" in value
              for value in context["budget"].values()), context["budget"])

    poisoned = series("SPY") + [{"date": "2026-08-12", "close": 9999,
                                  "high": 9999, "low": 9999, "volume": 1}]
    safe = market_features(poisoned, cutoff)
    check("future poison ignored", safe["close"] != 9999, safe)
    print("ALL PASS")


if __name__ == "__main__":
    main()
