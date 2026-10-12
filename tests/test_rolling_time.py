"""Regression checks for point-in-time rolling timestamps.

Run: .venv/bin/python tests/test_rolling_time.py
"""
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.rolling_time import (cutoff_for, is_trading_day, last_n_trading_days,  # noqa: E402
                              parse_event_et, parse_execution_et, trading_days)


def main():
    assert parse_execution_et("2026-08-03T01:30:00").isoformat() == \
        "2026-08-02T21:30:00-04:00"
    assert parse_event_et("2026-08-03T17:15:00-04:00").hour == 17
    assert cutoff_for(date(2026, 8, 12)).isoformat() == \
        "2026-08-11T16:00:00-04:00"
    assert [d.isoformat() for d in last_n_trading_days(date(2026, 8, 11), 7)] == [
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
        "2026-08-07", "2026-08-10", "2026-08-11",
    ]
    horizon = trading_days(date(2026, 8, 3), date(2026, 10, 7))[7:47]
    assert len(horizon) == 40 and horizon[0] == date(2026, 8, 12)
    assert horizon[-1] == date(2026, 10, 7)
    assert all(is_trading_day(d) and d.weekday() < 5 for d in horizon)
    assert date(2026, 9, 7) not in horizon  # Labor Day exchange holiday
    print("PASS rolling time contract")


if __name__ == "__main__":
    main()
