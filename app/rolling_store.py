"""Immutable, idempotent storage for point-in-time profile snapshots."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


class RollingProfileStore:
    """One immutable logical profile version per user and cutoff."""

    def __init__(self, db_path):
        self.db_path = str(db_path)

    def _con(self):
        c = sqlite3.connect(self.db_path, timeout=30)
        c.execute("pragma journal_mode=WAL")
        c.execute("pragma busy_timeout=30000")
        c.row_factory = sqlite3.Row
        c.execute("""create table if not exists rolling_profile_versions(
            id integer primary key autoincrement,
            user_id text not null,
            cutoff_et text not null,
            version integer not null,
            parent_version integer,
            profile_json text not null,
            run_json text not null,
            created_at text not null,
            unique(user_id, cutoff_et),
            unique(user_id, version))""")
        return c

    def latest_before(self, user_id, cutoff_et=None):
        c = self._con()
        where = "user_id=?"
        params = [str(user_id)]
        if cutoff_et is not None:
            where += " and cutoff_et<?"
            params.append(str(cutoff_et))
        row = c.execute(
            f"select * from rolling_profile_versions where {where} "
            "order by cutoff_et desc,version desc limit 1", params).fetchone()
        return dict(row) if row else None

    def get(self, user_id, cutoff_et):
        c = self._con()
        row = c.execute(
            "select * from rolling_profile_versions where user_id=? and cutoff_et=?",
            (str(user_id), str(cutoff_et))).fetchone()
        return dict(row) if row else None

    def save(self, user_id, cutoff_et, profile, run_meta):
        """Save once; an identical retry returns the existing version.

        A retry with different content is rejected instead of silently rewriting an
        experiment's historical input.
        """
        uid, cutoff = str(user_id), str(cutoff_et)
        profile_blob = json.dumps(profile, ensure_ascii=False, sort_keys=True)
        run_blob = json.dumps(run_meta or {}, ensure_ascii=False, sort_keys=True)
        c = self._con()
        c.execute("begin immediate")
        existing = c.execute(
            "select version,profile_json,run_json from rolling_profile_versions "
            "where user_id=? and cutoff_et=?", (uid, cutoff)).fetchone()
        if existing:
            c.rollback()
            if existing["profile_json"] != profile_blob:
                raise ValueError(f"immutable rolling profile already exists for {uid} at {cutoff}")
            return int(existing["version"]), False
        parent = c.execute(
            "select version from rolling_profile_versions where user_id=? "
            "and cutoff_et<? order by cutoff_et desc limit 1", (uid, cutoff)).fetchone()
        parent_version = int(parent["version"]) if parent else None
        version = (c.execute(
            "select coalesce(max(version),0)+1 from rolling_profile_versions where user_id=?",
            (uid,)).fetchone()[0])
        c.execute(
            "insert into rolling_profile_versions(user_id,cutoff_et,version,parent_version,"
            "profile_json,run_json,created_at) values(?,?,?,?,?,?,?)",
            (uid, cutoff, version, parent_version, profile_blob, run_blob,
             datetime.now(timezone.utc).isoformat()))
        c.commit()
        return int(version), True
