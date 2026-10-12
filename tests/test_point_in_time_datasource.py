"""Run: .venv/bin/python tests/test_point_in_time_datasource.py"""
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.datasource import LinkBMDataSource  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "source.db"
        c = sqlite3.connect(db)
        c.executescript("""
        create table events(user_id text,event_date text,timestamp_et text,action_text text,source text);
        create table bm452_trades(user_id text,trade_date text,ticker text,side text,
          executed_quantity real,executed_amount real,execution_state text,payload text,
          executed_at text,order_at text,venue text);
        create table bm453_daily(user_id text,snapshot_date text,asset_total real,
          stock_total real,money_remain real,stock_profit_daily real,holdings_json text);
        create table bm428_conversations(user_id text,seq integer,title text);
        """)
        c.executemany("insert into events values(?,?,?,?,?)", [
            ("u", "2026-08-11", "2026-08-11T15:59:59-04:00", "before", "x"),
            ("u", "2026-08-11", "2026-08-11T16:00:01-04:00", "after", "x"),
        ])
        c.executemany("insert into bm452_trades values(?,?,?,?,?,?,?,?,?,?,?)", [
            ("u", "wrong", "A", "buy", 1, 10, "executed", None,
             "2026-08-11T19:59:59", None, None),
            ("u", "wrong", "B", "buy", 1, 10, "executed", None,
             "2026-08-11T20:00:01", None, None),
        ])
        c.execute("insert into bm453_daily values(?,?,?,?,?,?,?)",
                  ("u", "2026-08-11", 10, 0, 10, 0, "{}"))
        c.execute("insert into bm428_conversations values('u',1,'undated')")
        c.commit(); c.close()
        ds = LinkBMDataSource(db).at("2026-08-11T16:00:00-04:00")
        assert [x["action_text"] for x in ds.get_behaviors("u")] == ["before"]
        assert [x["ticker"] for x in ds.get_trades("u")] == ["A"]
        assert ds.get_trades("u")[0]["trade_day"] == "2026-08-11"
        assert len(ds.get_account_daily("u")) == 1
        assert ds.get_conversations("u") == []
    print("PASS point-in-time datasource")


if __name__ == "__main__":
    main()
