#!/usr/bin/env python3
"""Parse archived BM452/BM453/BM433 payloads into normalised tables.

BM452 payload is a `|`-separated list of `#`-separated order records. The field
layout below was verified in the sibling repo on 359 rows; this script re-verifies
it on the full corpus and FAILS LOUDLY if the layout does not hold, because a
silent mis-index produces fabricated timestamps.

CRITICAL: BM452 carries no usable trade date column. The trailing `settlement
date` (bit 23) is the label-extraction bag date, and using it makes every
time-window metric measure the extraction batch instead of the user. The true
execution time is bit 18 (epoch ms). We keep both, decode from bit 18, and report
the disagreement rate as evidence.

Outputs
  data/parsed/bm452_trades.jsonl   one decoded order per line
  data/parsed/bm453_daily.jsonl    one account-day per line
  data/parsed/bm433_events.jsonl   normalised behaviour events (Beijing -> ET)
  data/parsed/user_presence.json   per-user stream presence + volumes
  data/bm.db                       sqlite mirror
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw_bm"
PARSED = DATA / "parsed"

BEIJING = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")
CODES = ("BM433", "BM452", "BM453")

# --- BM452 field layout (index -> meaning) -----------------------------------
F_ORDER_ID, F_BROKER, F_USER, F_ACCOUNT = 0, 1, 2, 3
F_INSTRUMENT_UID, F_TICKER, F_INSTRUMENT_NAME = 4, 5, 6
F_ORDER_QTY, F_CANCELLED_QTY, F_EXECUTED_QTY, F_EXECUTED_AMOUNT = 10, 11, 12, 13
F_SIDE = 14
F_EXECUTED_AT_MS = 18
F_ACK_AT_MS = 19
F_VENUE = 21
F_ORDER_AT_MS = 22
F_BAG_DATE = 23
MIN_BITS = 24
SIDE_MAP = {"1001": "buy", "1002": "sell"}
EPOCH_MIN, EPOCH_MAX = 1_600_000_000.0, 1_900_000_000.0  # 2020-09 .. 2030-03

# --- BM433 text layout -------------------------------------------------------
DATE_MARKER = re.compile(r"\*{1,2}(\d{4}-\d{2}-\d{2})")
ACTION_MARKER = re.compile(r"\*(\d{2}:\d{2})\s*(.*?)(?=\*\d{2}:\d{2}\s|$)", re.S)
TIMEFRAME_SWITCH = re.compile(r"^switch to (.+?) timeframe$", re.I)


def num(v):
    try:
        f = float(v)
        return None if f != f else f          # nan guard
    except (TypeError, ValueError):
        return None


def epoch_ms(v):
    """bit -> naive UTC datetime, or None when the bit is not a valid epoch."""
    if not (v or "").isdigit() or len(v) != 13:
        return None
    ts = int(v) / 1000.0
    if not (EPOCH_MIN < ts < EPOCH_MAX):
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def trading_date(d):
    """Naive-UTC execution stamp -> the America/New_York session date.

    `executed_at` is naive UTC. Its UTC calendar date is NOT the trading day:
    an after-hours fill at 20:30 ET is 00:30 UTC the next day, so the UTC date
    lands one session late. That mislabelled 999/16,953 rows (5.9%) and every
    one of them created a false negative on its real session and a false
    positive on the following one. Executions peak in the 09:00 ET hour (the
    open) and decay through the day; on the UTC clock that peak sits at 13:00.
    """
    if d is None:
        return None
    return d.replace(tzinfo=timezone.utc).astimezone(NEW_YORK).date().isoformat()


# --------------------------- BM452 ------------------------------------------
def parse_bm452(raw: str, stats: dict):
    if not raw:
        return []
    out = []
    for rec in raw.split("|"):
        # 98.7% of records carry surrounding whitespace in order_id/ticker.
        # Stripping is required, not cosmetic: unstripped ids break joins and
        # made `(user_id, order_id)` collapse 100 DISTINCT rows in sqlite.
        raw_bits = rec.split("#")
        bits = [b.strip() for b in raw_bits]
        if any(a != b for a, b in zip(raw_bits, bits)):
            stats["records_whitespace_normalised"] += 1
        if len(bits) < MIN_BITS:
            stats["records_dropped_short"] += 1
            continue
        stats["records_total"] += 1
        side = SIDE_MAP.get(bits[F_SIDE])
        if side is None:
            stats["side_unmapped"] += 1
            stats[f"side_code_{bits[F_SIDE]}"] += 1
        executed_at = epoch_ms(bits[F_EXECUTED_AT_MS])
        if executed_at is None:
            stats["executed_at_undecodable"] += 1
        else:
            stats["executed_at_ok"] += 1
        bag = bits[F_BAG_DATE]
        bag_date = f"{bag[:4]}-{bag[4:6]}-{bag[6:]}" if re.fullmatch(r"\d{8}", bag) else None
        # The trap: does the bag date equal the real execution date?
        if executed_at and bag_date:
            stats["bag_vs_exec_compared"] += 1
            if bag_date != executed_at.date().isoformat():
                stats["bag_vs_exec_mismatch"] += 1
        # bit 11 = cancelled qty, bit 12 = executed qty.
        # NOTE: an earlier version swapped these two names and inverted every
        # execution_state, which silently reclassified 12,425 real executions as
        # cancellations and destroyed the buy/sell label set.
        cancelled_qty = num(bits[F_CANCELLED_QTY])
        executed_qty = num(bits[F_EXECUTED_QTY])
        state = ("executed" if (executed_qty or 0) != 0
                 else "cancelled_unfilled" if (cancelled_qty or 0) != 0
                 else "unfilled_or_unknown")
        out.append({
            "order_id": bits[F_ORDER_ID], "user_id": bits[F_USER],
            "ticker": bits[F_TICKER] or None, "instrument_name": bits[F_INSTRUMENT_NAME] or None,
            "side": side, "side_code": bits[F_SIDE],
            "order_qty": num(bits[F_ORDER_QTY]),
            "cancelled_qty": cancelled_qty,
            "executed_qty": executed_qty,
            "executed_amount": num(bits[F_EXECUTED_AMOUNT]),
            "order_type": bits[15] or None, "order_status": bits[17] or None,
            "executed_at": executed_at.isoformat() if executed_at else None,
            "executed_date": trading_date(executed_at),
            "order_at": (lambda d: d.isoformat() if d else None)(epoch_ms(bits[F_ORDER_AT_MS])),
            "venue": bits[F_VENUE] or None,
            "bag_date": bag_date,
            "execution_state": state,
        })
    return out


# --------------------------- BM453 ------------------------------------------
def parse_bm453(raw: str):
    if not raw:
        return []
    try:
        doc = json.loads(raw)
    except Exception:                                              # noqa: BLE001
        return []
    if not isinstance(doc, dict):
        return []
    rows = []
    for day, payload in sorted(doc.items()):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day)) or not isinstance(payload, dict):
            continue
        holdings = payload.get("holdings") or {}
        clean = {}
        if isinstance(holdings, dict):
            for code, h in holdings.items():
                if not isinstance(h, dict):
                    continue
                clean[code] = {
                    "stock_name": h.get("stock_name"),
                    "stock_vol": num(h.get("stock_vol")),
                    "last_price": num(h.get("last_price")),
                    "cost_price": num(h.get("cost_price")),
                    "stock_market_value": num(h.get("stock_market_value")),
                    "profit_loss_total": num(h.get("profit_loss_total")),
                }
        rows.append({
            "snapshot_date": day,
            "asset_total": num(payload.get("asset_total")),
            "stock_total": num(payload.get("stock_total")),
            "money_remain": num(payload.get("money_remain")),
            "stock_profit_daily": num(payload.get("stock_profit_daily")),
            "n_holdings": len(clean),
            "holdings": clean,
        })
    return rows


# --------------------------- BM433 ------------------------------------------
def parse_bm433(raw: str):
    """BM433 text -> events. Source stamps are Beijing time, converted to ET."""
    if not raw or not isinstance(raw, str):
        return []
    events = []
    markers = list(DATE_MARKER.finditer(raw))
    for i, marker in enumerate(markers):
        day = marker.group(1)
        end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        for act in ACTION_MARKER.finditer(raw[marker.end():end]):
            text = act.group(2).strip().strip('"')
            if not text:
                continue
            try:
                bj = datetime.fromisoformat(f"{day}T{act.group(1)}:00").replace(tzinfo=BEIJING)
            except ValueError:
                continue
            et = bj.astimezone(NEW_YORK)
            tf = TIMEFRAME_SWITCH.match(text)
            events.append({
                "timestamp_bjt": bj.isoformat(), "timestamp_et": et.isoformat(),
                "event_date": et.date().isoformat(), "event_time_et": et.strftime("%H:%M"),
                "action_text": text,
                "timeframe": tf.group(1) if tf else None,
            })
    events.sort(key=lambda e: e["timestamp_et"])
    return events


def main() -> int:
    PARSED.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.glob("*.json") if p.name != "manifest.json")
    print(f"parsing {len(files)} raw files\n")

    stats = Counter()
    presence = {}
    all_trades, all_daily, all_events = [], [], []

    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  SKIP {path.name}: {exc}")
            continue
        uid = str(doc.get("user_id") or path.stem)
        trades = parse_bm452(doc.get("BM452") or "", stats)
        daily = parse_bm453(doc.get("BM453") or "")
        events = parse_bm433(doc.get("BM433") or "")
        for t in trades:
            t["user_id"] = uid
            all_trades.append(t)
        for d in daily:
            d["user_id"] = uid
            all_daily.append(d)
        for e in events:
            e["user_id"] = uid
            all_events.append(e)

        trade_days = {t["executed_date"] for t in trades if t["executed_date"]}
        presence[uid] = {
            "BM433": len(events) > 0, "BM452": len(trades) > 0, "BM453": len(daily) > 0,
            "n_events": len(events), "n_trades": len(trades), "n_days": len(daily),
            "n_holdings_last_day": next((d["n_holdings"] for d in reversed(daily)), 0),
            "n_trade_days": len(trade_days),
            "trade_first": min(trade_days) if trade_days else None,
            "trade_last": max(trade_days) if trade_days else None,
            "daily_first": daily[0]["snapshot_date"] if daily else None,
            "daily_last": daily[-1]["snapshot_date"] if daily else None,
            "bytes": {c: len(doc.get(c) or "") for c in CODES},
        }

    def dump(name, rows):
        p = PARSED / name
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    dump("bm452_trades.jsonl", all_trades)
    dump("bm453_daily.jsonl", all_daily)
    dump("bm433_events.jsonl", all_events)

    # ---- sqlite mirror -----------------------------------------------------
    db = DATA / "bm.db"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute("pragma journal_mode=WAL")
    con.executescript("""
    -- Surrogate PK on purpose. `(user_id, order_id)` is NOT unique: 55 keys
    -- carried 100 distinct rows (partial fills / repeated orders). A composite
    -- PK silently deleted them, which is exactly the kind of lossy dedup that
    -- corrupts trade counts.
    create table bm452_trades(
      row_pk integer primary key autoincrement,
      user_id text, order_id text, executed_at text, executed_date text,
      ticker text, side text, side_code text, order_qty real, executed_qty real,
      executed_amount real, execution_state text, bag_date text);
    create table bm453_daily(
      user_id text, snapshot_date text, asset_total real, stock_total real,
      money_remain real, stock_profit_daily real, n_holdings integer,
      holdings_json text, primary key(user_id, snapshot_date));
    create table bm433_events(
      user_id text, timestamp_et text, event_date text, event_time_et text,
      action_text text, timeframe text);
    create index ix_trades on bm452_trades(user_id, executed_date);
    create index ix_trades_oid on bm452_trades(user_id, order_id);
    create index ix_daily  on bm453_daily(user_id, snapshot_date);
    create index ix_events on bm433_events(user_id, timestamp_et);
    """)
    con.executemany(
        "insert into bm452_trades(user_id,order_id,executed_at,executed_date,ticker,"
        "side,side_code,order_qty,executed_qty,executed_amount,execution_state,bag_date)"
        " values(?,?,?,?,?,?,?,?,?,?,?,?)",
        [(t["user_id"], t["order_id"], t["executed_at"], t["executed_date"], t["ticker"],
          t["side"], t["side_code"], t["order_qty"], t["executed_qty"],
          t["executed_amount"], t["execution_state"], t["bag_date"]) for t in all_trades])
    con.executemany(
        "insert or replace into bm453_daily values(?,?,?,?,?,?,?,?)",
        [(d["user_id"], d["snapshot_date"], d["asset_total"], d["stock_total"],
          d["money_remain"], d["stock_profit_daily"], d["n_holdings"],
          json.dumps(d["holdings"], ensure_ascii=False)) for d in all_daily])
    con.executemany(
        "insert into bm433_events values(?,?,?,?,?,?)",
        [(e["user_id"], e["timestamp_et"], e["event_date"], e["event_time_et"],
          e["action_text"], e["timeframe"]) for e in all_events])
    con.commit()
    con.close()

    both = sorted(u for u, p in presence.items() if all(p[c] for c in CODES))
    (PARSED / "user_presence.json").write_text(json.dumps({
        "codes": list(CODES),
        "counts": {c: sum(1 for p in presence.values() if p[c]) for c in CODES},
        "all_three": len(both),
        "all_three_user_ids": both,
        "users": presence,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- report ------------------------------------------------------------
    print(f"BM452 orders parsed      : {stats['records_total']:,} "
          f"(short/dropped {stats['records_dropped_short']})")
    print(f"  whitespace-normalised  : {stats['records_whitespace_normalised']:,} "
          f"({100*stats['records_whitespace_normalised']/max(stats['records_total'],1):.1f}%)")
    print(f"  executed_at decodable  : {stats['executed_at_ok']:,} "
          f"({100*stats['executed_at_ok']/max(stats['records_total'],1):.2f}%)")
    print(f"  side unmapped          : {stats['side_unmapped']:,}")
    mv = stats["bag_vs_exec_mismatch"]; cmp_ = stats["bag_vs_exec_compared"]
    print(f"  bag-date != exec-date  : {mv:,}/{cmp_:,} "
          f"({100*mv/max(cmp_,1):.1f}%)  <- why bag_date must never be used as trade_date")
    print(f"BM453 account-days       : {len(all_daily):,}")
    print(f"BM433 events             : {len(all_events):,}")
    print()
    for c in CODES:
        print(f"  {c}: {sum(1 for p in presence.values() if p[c])} users")
    print(f"  ALL THREE: {len(both)} users")
    print(f"\nwrote {PARSED}/*.jsonl, data/bm.db, data/parsed/user_presence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
