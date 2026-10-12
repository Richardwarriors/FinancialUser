"""Profile store abstraction.

v1 persists to SQLite (profile_versions table, already in use).
A MongoProfileStore / remote implementation can replace this behind the same
interface once the MongoDB decision lands.
"""
import json
import sqlite3
from datetime import datetime, timezone


class ProfileStore:
    def save(self, user_id, profile, evidence_refs, meta):
        raise NotImplementedError

    def update_latest(self, user_id, profile):
        """Rewrite the newest version's payload WITHOUT creating a version.

        Returns the version number updated, or None when the user has no row.
        """
        raise NotImplementedError

    def prune_history(self, keep_per_user=1):
        """Delete all but the newest `keep_per_user` versions per user.

        Returns {"deleted": n, "kept": n, "users": n}. Version numbers are NOT
        renumbered: a version id is an audit reference, so gaps are honest and
        renumbering would silently repoint existing references.
        """
        raise NotImplementedError


class SQLiteProfileStore(ProfileStore):
    """Writes to profile_versions (business history), same collection shape as
    the planned Mongo current/history split: latest row = current."""

    def __init__(self, db_path):
        self.db_path = str(db_path)

    def save(self, user_id, profile, evidence_refs, meta):
        c = sqlite3.connect(self.db_path)
        c.execute("""create table if not exists profile_versions(
            id integer primary key autoincrement,user_id text,version integer,
            profile_json text,evidence_json text,created_at text default current_timestamp)""")
        n = c.execute("select coalesce(max(version),0)+1 from profile_versions where user_id=?",
                      (user_id,)).fetchone()[0]
        # `version` is written into the document as well as the column: the
        # lifecycle layer needs to say WHICH version a carried-forward value came
        # from, and load_existing only reads the document.
        doc = {**profile, "version": n,
               "_meta": {**meta, "saved_at": datetime.now(timezone.utc).isoformat()}}
        c.execute("insert into profile_versions(user_id,version,profile_json,evidence_json) values(?,?,?,?)",
                  (user_id, n, json.dumps(doc, ensure_ascii=False), json.dumps(evidence_refs, ensure_ascii=False)))
        c.commit()
        return n

    def get_latest(self, user_id):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("""create table if not exists profile_versions(
            id integer primary key autoincrement,user_id text,version integer,
            profile_json text,evidence_json text,created_at text default current_timestamp)""")
        r = c.execute("select * from profile_versions where user_id=? order by version desc limit 1",
                      (user_id,)).fetchone()
        return dict(r) if r else None

    def update_latest(self, user_id, profile):
        """Refresh the derived layer of the newest version, in place.

        Used when the prose layer (portrait / KYC tree / profile tree) is rebuilt
        while the five dimension values are byte-identical. Writing a NEW version
        for that would fill the version history with entries whose business
        content did not change, and "what changed, and when" is the only question
        the version history exists to answer — so the version number must track
        business change, not presentation rebuilds.

        The `evidence_json` column and the row's `created_at` are preserved: the
        evidence behind the dimensions genuinely did not change, and `created_at`
        dates the version's business content, which is also unchanged.
        """
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        r = c.execute("select id, version, profile_json from profile_versions "
                      "where user_id=? order by version desc limit 1",
                      (user_id,)).fetchone()
        if r is None:
            return None
        prior = json.loads(r["profile_json"] or "{}")
        # Keep the stored version number and _meta stamp of the row we are
        # rewriting, then overlay the freshly built derived blocks.
        doc = {**prior, **profile,
               "version": r["version"],
               "_meta": {**(prior.get("_meta") or {}),
                         "derived_refreshed_at": datetime.now(timezone.utc).isoformat()}}
        c.execute("update profile_versions set profile_json=? where id=?",
                  (json.dumps(doc, ensure_ascii=False), r["id"]))
        c.commit()
        return int(r["version"])

    def prune_history(self, keep_per_user=1, user_ids=None):
        """Keep only the newest `keep_per_user` rows per user.

        Implemented as a delete keyed on rowid so it works on SQLite versions
        without window-function support. Use ids, not versions: versions restart
        per user, so the newest row is not globally the largest version.
        """
        if keep_per_user < 0:
            raise ValueError("keep_per_user must be >= 0")
        c = sqlite3.connect(self.db_path)
        where, params = "", []
        if user_ids:
            where = f" where user_id in ({','.join('?' * len(user_ids))})"
            params = list(user_ids)
        rows = c.execute(
            f"select user_id, version from profile_versions{where} "
            "order by user_id, version desc", params).fetchall()
        keep, seen = [], {}
        for user_id, version in rows:
            n = seen.get(user_id, 0)
            if n < keep_per_user:
                seen[user_id] = n + 1
                keep.append((user_id, version))
        keep_set = set(keep)
        doomed = [(u, v) for u, v in rows if (u, v) not in keep_set]
        if doomed:
            c.executemany("delete from profile_versions where user_id=? and version=?", doomed)
        c.commit()
        total = c.execute("select count(*) from profile_versions").fetchone()[0]
        return {"deleted": len(doomed), "kept": total, "users": len(seen)}
