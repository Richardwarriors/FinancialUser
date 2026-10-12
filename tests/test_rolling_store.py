"""Run: .venv/bin/python tests/test_rolling_store.py"""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.rolling_store import RollingProfileStore  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as td:
        store = RollingProfileStore(Path(td) / "test.db")
        v1, made = store.save("u1", "2026-08-01T16:00:00-04:00", {"x": 1}, {})
        assert (v1, made) == (1, True)
        assert store.save("u1", "2026-08-01T16:00:00-04:00", {"x": 1}, {}) == (1, False)
        try:
            store.save("u1", "2026-08-01T16:00:00-04:00", {"x": 2}, {})
        except ValueError:
            pass
        else:
            raise AssertionError("immutable row was overwritten")
        assert store.save("u1", "2026-08-02T16:00:00-04:00", {"x": 2}, {}) == (2, True)
        latest = store.latest_before("u1", "2026-08-03T00:00:00-04:00")
        assert json.loads(latest["profile_json"]) == {"x": 2}
    print("PASS immutable rolling profile store")


if __name__ == "__main__":
    main()
