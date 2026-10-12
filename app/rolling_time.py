"""Point-in-time helpers for the rolling prediction dataset.

All execution timestamps are normalised to America/New_York before they are
assigned to a trading day.  The BM452 ``executed_date`` column is deliberately
not used: it is not a reliable exchange-local date.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_CLOSE = time(16, 0)


def _observed(y: int, m: int, d: int) -> date:
    value = date(y, m, d)
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth(y: int, m: int, weekday: int, n: int) -> date:
    value = date(y, m, 1)
    return value + timedelta(days=(weekday - value.weekday()) % 7 + 7 * (n - 1))


def _last(y: int, m: int, weekday: int) -> date:
    value = (date(y, m + 1, 1) - timedelta(days=1)
             if m < 12 else date(y, 12, 31))
    return value - timedelta(days=(value.weekday() - weekday) % 7)


def holidays(y: int) -> set[date]:
    """US equity-market full-day holidays used by this 2026 benchmark."""
    return {
        _observed(y, 1, 1), _nth(y, 1, 0, 3), _nth(y, 2, 0, 3),
        _last(y, 5, 0), _observed(y, 6, 19), _observed(y, 7, 4),
        _nth(y, 9, 0, 1), _nth(y, 11, 3, 4), _observed(y, 12, 25),
    }


def is_trading_day(value: date) -> bool:
    return value.weekday() < 5 and value not in holidays(value.year)


def trading_days(start: date, end: date) -> list[date]:
    out = []
    current = start
    while current <= end:
        if is_trading_day(current):
            out.append(current)
        current += timedelta(days=1)
    return out


def previous_trading_day(value: date) -> date:
    current = value - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current


def cutoff_for(target: date) -> datetime:
    return datetime.combine(previous_trading_day(target), MARKET_CLOSE,
                            tzinfo=NEW_YORK)


def parse_event_et(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NEW_YORK)
    return parsed.astimezone(NEW_YORK)


def parse_execution_et(value: str | None) -> datetime | None:
    """Parse BM452 ``executed_at`` (stored as naive UTC) into aware ET."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(NEW_YORK)


def parse_cutoff(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid cutoff timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("cutoff timestamp must include a timezone")
    return parsed.astimezone(NEW_YORK)


def last_n_trading_days(cutoff: date, n: int) -> list[date]:
    if n < 1:
        return []
    out = []
    current = cutoff
    while len(out) < n:
        if is_trading_day(current):
            out.append(current)
        current -= timedelta(days=1)
    return list(reversed(out))
