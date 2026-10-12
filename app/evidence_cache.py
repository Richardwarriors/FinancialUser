"""Evidence cache: skip recomputation when the inputs are unchanged.

Measured cost of the deterministic layer for one core user (3,713 behavior_events events,
256 trades, 90 account days):

    fetch      0.09s
    intent     0.27s   <-- recomputed from every event on every refresh
    evidence   0.17s
    -------------------
    total      0.53s

At 12 demo users that is irrelevant, which is why this was deferred. At 200k users
it is ~29 CPU-hours per full sweep, so a cache is worth having — but only if the
hit path is genuinely cheaper than recomputation and cannot go stale.

Design constraints
------------------
* **Keyed by CONTENT, not by "did new rows arrive".** `source_hash` covers the
  whole snapshot, so a corrected value invalidates the entry exactly like new
  data does. Counting rows would serve stale evidence after a data repair.
* **Versioned.** `feature_version` folds in the engine versions, so a change to a
  metric definition invalidates every entry even when the inputs are identical.
* **Fail-open.** Any error (missing table, corrupt row, IO problem) falls through
  to recomputation. A cache must never be able to break profile generation.
* **Bounded.** Rows are pruned so the cache cannot grow without limit.
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone

CACHE_VERSION = "evidence-cache-v1"
# how many cached rows to retain; the eviction is by recency, not per user
MAX_ROWS = int(os.environ.get("PROFILE_EVIDENCE_CACHE_MAX_ROWS", "20000"))
ENABLED = os.environ.get("PROFILE_EVIDENCE_CACHE", "1") not in ("0", "false", "False")


def source_hash(snapshot):
    """Content hash of the raw snapshot.

    Covers everything the deterministic layer reads, so any change — appended
    events, a corrected `asset_total`, a different extraction — produces a new
    hash and misses the cache.
    """
    if not isinstance(snapshot, dict):
        return None
    parts = []
    for key in sorted(snapshot):
        if key in ("as_of", "extraction_time"):     # volatile, not inputs
            continue
        v = snapshot[key]
        blob = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
        parts.append(f"{key}:{hashlib.sha1(blob.encode('utf-8')).hexdigest()}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _con(db_path):
    c = sqlite3.connect(db_path, timeout=30)
    c.execute("pragma journal_mode=WAL")
    c.execute("pragma busy_timeout=30000")
    c.row_factory = sqlite3.Row
    return c


def _ensure(c):
    c.execute("""create table if not exists evidence_cache(
        user_id text, source_hash text, feature_version text,
        evidence_json text, created_at text,
        primary key (user_id, source_hash, feature_version))""")


def get(db_path, user_id, src_hash, feature_version):
    """Cached evidence, or None. Never raises."""
    if not ENABLED or not src_hash:
        return None
    try:
        c = _con(db_path)
        _ensure(c)
        r = c.execute("select evidence_json from evidence_cache where user_id=? "
                      "and source_hash=? and feature_version=?",
                      (user_id, src_hash, feature_version)).fetchone()
        if not r:
            return None
        return json.loads(r["evidence_json"])
    except Exception:                                                 # noqa: BLE001
        return None


def put(db_path, user_id, src_hash, feature_version, evidence):
    """Store evidence. Never raises; a failed write only costs a recomputation."""
    if not ENABLED or not src_hash:
        return False
    try:
        c = _con(db_path)
        _ensure(c)
        c.execute("insert or replace into evidence_cache"
                  "(user_id,source_hash,feature_version,evidence_json,created_at)"
                  " values(?,?,?,?,?)",
                  (user_id, src_hash, feature_version,
                   json.dumps(evidence, ensure_ascii=False),
                   datetime.now(timezone.utc).isoformat()))
        # bound the cache: keep the newest N rows
        total = c.execute("select count(*) from evidence_cache").fetchone()[0]
        if total > MAX_ROWS:
            c.execute("delete from evidence_cache where rowid in "
                      "(select rowid from evidence_cache order by created_at asc limit ?)",
                      (total - MAX_ROWS,))
        c.commit()
        return True
    except Exception:                                                 # noqa: BLE001
        return False


def stats(db_path):
    try:
        c = _con(db_path)
        _ensure(c)
        n = c.execute("select count(*) from evidence_cache").fetchone()[0]
        u = c.execute("select count(distinct user_id) from evidence_cache").fetchone()[0]
        return {"enabled": ENABLED, "version": CACHE_VERSION,
                "rows": n, "users": u, "max_rows": MAX_ROWS}
    except Exception as e:                                            # noqa: BLE001
        return {"enabled": ENABLED, "error": f"{type(e).__name__}: {e}"}


def clear(db_path):
    c = _con(db_path)
    _ensure(c)
    n = c.execute("select count(*) from evidence_cache").fetchone()[0]
    c.execute("delete from evidence_cache")
    c.commit()
    return n
