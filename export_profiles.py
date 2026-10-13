#!/usr/bin/env python3
"""Export stored T0 profiles using the rolling prediction's strict schema."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from app.rolling_config import PROFILE_CUTOFF
from app.rolling_profile import export_profile

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(HERE / "data" / "observations_t0.db"))
    ap.add_argument("--out", default=str(HERE / "data" / "rolling_prediction" / "base_t0"))
    ap.add_argument("--users-file")
    ap.add_argument("--expected-users", type=int, default=120)
    args = ap.parse_args()
    wanted = set(Path(args.users_file).read_text().split()) if args.users_file else None
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = c.execute("select user_id,version,profile_json from profile_versions "
                     "order by user_id,version desc").fetchall()
    latest = {}
    for uid, version, raw in rows:
        if wanted is not None and uid not in wanted:
            continue
        latest.setdefault(uid, (version, raw))
    if len(latest) != args.expected_users:
        raise SystemExit(f"expected {args.expected_users} profiles, found {len(latest)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cutoff = f"{PROFILE_CUTOFF}T16:00:00-04:00"
    manifest = {"schema_version": "financial_t0_manifest_v1", "cutoff_et": cutoff,
                "expected_users": args.expected_users, "profiles": []}
    for uid, (version, raw) in sorted(latest.items()):
        profile = json.loads(raw)
        doc = export_profile(uid, profile, cutoff, version=version,
                             run_meta={"source": args.db, "kind": "fixed_t0"})
        path = out / f"{uid}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        manifest["profiles"].append({"user_id": uid, "file": path.name,
                                     "profile_version": version})
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(latest)} validated T0 profiles to {out}")


if __name__ == "__main__":
    main()
