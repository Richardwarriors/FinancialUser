#!/usr/bin/env python3
"""Build a NAACL27 profile database from the raw BM archive, truncated at T0.

Why this exists instead of `migrate_new_bm.py`:

  1. its loader takes `bits[-1]` as the trade date, but that field is the label
     extraction bag date and disagrees with the true execution date on 98.6% of
     rows (verified on 348 rows: 348/348 mismatched);
  2. it writes every row as `execution_state='executed'`, discarding cancellations;
  3. neither it nor the datasource applies any time boundary, so a profile built
     from the production layout silently absorbs post-T0 evidence.

This loader reuses the verified decoder (`parse_bm_payloads.py`) and enforces the
truncation at write time, so a downstream consumer that reads the whole table still
only sees pre-T0 rows.

Usage:
    python3 load_naacl_db.py --out data/bm_t0.db --cutoff 2026-08-01
    python3 load_naacl_db.py --out data/bm_t0.db --cutoff 2026-08-01 --users-file data/users_core_120.txt
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Reuse the verified decoder rather than re-deriving the bit layout.
sys.path.insert(0, str(HERE / "vendor"))
from parse_bm_payloads import parse_bm452, parse_bm453, parse_bm433

SCHEMA = """
create table if not exists users(
    user_id text primary key, base_profile_summary text, source_file text);
create table if not exists events(
    id integer primary key autoincrement, user_id text, event_date text,
    timestamp_et text, action_text text, source text);
create table if not exists bm452_trades(
    id integer primary key autoincrement, user_id text, trade_date text,
    ticker text, side text, executed_quantity real, executed_amount real,
    execution_state text, payload text, source text,
    executed_at text, order_at text, venue text);
create table if not exists bm453_daily(
    user_id text not null, snapshot_date text not null,
    asset_total real, stock_total real, money_remain real,
    stock_profit_daily real, holdings_json text, source text,
    primary key(user_id, snapshot_date));
create table if not exists bm428_conversations(
    user_id text not null, seq integer not null, title text, source text,
    primary key(user_id, seq));
create table if not exists profile_versions(
    id integer primary key autoincrement, user_id text, version integer,
    profile_json text, evidence_json text, created_at text default current_timestamp);
create table if not exists profile_updates(
    id integer primary key autoincrement, user_id text, update_date text,
    payload text, source text);
create table if not exists evidence_cache(
    user_id text, source_hash text, feature_version text,
    evidence_json text, created_at text,
    primary key(user_id, source_hash, feature_version));
create table if not exists audit_log(
    id integer primary key autoincrement, task text, user_id text, status text,
    detail text, created_at text default current_timestamp);
create index if not exists ix_bm453_date on bm453_daily(user_id, snapshot_date);
create index if not exists ix_events_user on events(user_id, event_date);
create index if not exists ix_bm452_user on bm452_trades(user_id, trade_date);
create index if not exists ix_bm452_exec on bm452_trades(user_id, executed_at);
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "data" / "raw_bm"))
    ap.add_argument("--out", default=str(HERE / "data" / "bm_t0.db"))
    ap.add_argument("--cutoff", default="2026-08-01",
                    help="profile boundary T0; rows with date <= cutoff are loaded")
    ap.add_argument("--users-file", default=None,
                    help="optional allow-list of user ids, one per line")
    args = ap.parse_args()

    raw_dir, out_path = Path(args.raw), Path(args.out)
    cutoff = args.cutoff
    allow = None
    if args.users_file:
        allow = set(Path(args.users_file).read_text().split())
        print(f"restricting to {len(allow)} users from {args.users_file}")

    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(out_path)
    con.executescript(SCHEMA)

    files = sorted(p for p in raw_dir.glob("[0-9]*.json"))
    print(f"parsing {len(files)} raw files, cutoff = {cutoff}\n")

    stats = Counter()
    seen = set()
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                                    # noqa: BLE001
            print(f"  SKIP {path.name}: {exc}")
            continue
        uid = str(doc.get("user_id") or path.stem)
        if allow is not None and uid not in allow:
            continue
        seen.add(uid)

        # -- surface evidence (code) -----------------------------------------
        trade_stats = Counter()
        trades = parse_bm452(doc.get("BM452") or "", trade_stats)
        daily = parse_bm453(doc.get("BM453") or "")
        events = parse_bm433(doc.get("BM433") or "")

        # -- BM452: only rows on or before the cutoff ------------------------
        # Sign convention: the raw payload carries signed quantities (sells are
        # negative), but `trade_metrics.normalise_trades` drops any row with
        # `qty <= 0` and relies on `side` for direction. Storing the raw sign
        # would therefore discard every sell — 1,078 of 2,696 executed rows
        # (40%) — and silently break FIFO matching, realised P&L and the
        # disposition-effect signal. Store magnitudes, as the upstream loader
        # does; `side` is the direction.
        rows = []
        for t in trades:
            # `executed_date` is the America/New_York session date (see
            # parse_bm_payloads.trading_date). It was previously the UTC
            # calendar date, which put after-hours fills one session late and
            # therefore also compared them against the cutoff one day late.
            d = t.get("executed_date")
            if args.cutoff and d and d > cutoff:
                stats["bm452_after_cutoff"] += 1
                continue
            q_raw, a_raw = t.get("executed_qty"), t.get("executed_amount")
            rows.append((uid, d, t.get("ticker"), t.get("side"),
                         abs(q_raw) if q_raw is not None else None,
                         abs(a_raw) if a_raw is not None else None,
                         t.get("execution_state"), None, str(path),
                         t.get("executed_at"), t.get("order_at"), t.get("venue")))
        con.executemany(
            "insert into bm452_trades(user_id,trade_date,ticker,side,"
            "executed_quantity,executed_amount,execution_state,payload,source,"
            "executed_at,order_at,venue) values(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        stats["bm452_rows"] += len(rows)
        stats["bm452_executed"] += sum(1 for r in rows if r[6] == "executed")

        # -- BM453: only snapshots on or before the cutoff -------------------
        drows = []
        for d in daily:
            sd = d.get("snapshot_date")
            if args.cutoff and sd and sd > cutoff:
                stats["bm453_after_cutoff"] += 1
                continue
            drows.append((uid, sd, d.get("asset_total"), d.get("stock_total"),
                          d.get("money_remain"), d.get("stock_profit_daily"),
                          json.dumps(d.get("holdings") or {}, ensure_ascii=False),
                          str(path)))
        con.executemany(
            "insert or replace into bm453_daily(user_id,snapshot_date,asset_total,"
            "stock_total,money_remain,stock_profit_daily,holdings_json,source)"
            " values(?,?,?,?,?,?,?,?)", drows)
        stats["bm453_rows"] += len(drows)

        # -- BM433: only events on or before the cutoff ----------------------
        erows = []
        for e in events:
            ed = e.get("event_date")
            if args.cutoff and ed and ed > cutoff:
                stats["bm433_after_cutoff"] += 1
                continue
            erows.append((uid, ed, e.get("timestamp_et"), e.get("action_text"),
                          str(path)))
        con.executemany(
            "insert into events(user_id,event_date,timestamp_et,action_text,source)"
            " values(?,?,?,?,?)", erows)
        stats["bm433_rows"] += len(erows)

        con.execute("insert or replace into users(user_id,base_profile_summary,"
                    "source_file) values(?,?,?)", (uid, None, str(path)))

    con.commit()

    # ---- report ------------------------------------------------------------ #
    print(f"users loaded            : {len(seen)}")
    print(f"BM452 rows (<= {cutoff}) : {stats['bm452_rows']:,}"
          f"   executed {stats['bm452_executed']:,}"
          f"   (dropped {stats['bm452_after_cutoff']:,} after cutoff)")
    print(f"BM453 rows (<= {cutoff}) : {stats['bm453_rows']:,}"
          f"   (dropped {stats['bm453_after_cutoff']:,})")
    print(f"BM433 rows (<= {cutoff}) : {stats['bm433_rows']:,}"
          f"   (dropped {stats['bm433_after_cutoff']:,})")

    print("\nleakage check (no row may carry a date after the cutoff):")
    bad = 0
    for table, col in (("bm452_trades", "trade_date"), ("bm453_daily", "snapshot_date"),
                       ("events", "event_date")):
        n = con.execute(f"select count(*) from {table} where {col} > ?",
                        (cutoff,)).fetchone()[0]
        bad += n
        print(f"  {table:14} rows after cutoff: {n}")
    print(f"  -> {'CLEAN' if bad == 0 else 'LEAK DETECTED'}")

    con.close()
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
