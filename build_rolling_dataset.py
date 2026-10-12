#!/usr/bin/env python3
"""Build leakage-audited rolling samples and immutable daily profile versions.

The first target is the eighth trading day after T0.  Each sample contains an
expanding point-in-time profile, the preceding seven trading days of user data,
a compact market block, and two independently derived target contracts.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from app.datasource import LinkBMDataSource
from app.harness_graph import refresh
from app.naacl_config import PROFILE_CUTOFF
from app.rolling_labels import build_targets
from app.rolling_market import YahooMarketCache, build_market_context
from app.rolling_profile import (forecast_baseline, llm_restatement,
                                 persist_snapshot, public_profile)
from app.rolling_prompt import build_prompt_view
from app.rolling_recent import build_recent_7d
from app.rolling_schema import SAMPLE_SCHEMA_VERSION, validate_profile, validate_sample
from app.rolling_store import RollingProfileStore
from app.rolling_time import (NEW_YORK, cutoff_for,
                              parse_execution_et, trading_days)

HERE = Path(__file__).resolve().parent
DEFAULT_FULL_DB = HERE / "data" / "bm_full.db"
DEFAULT_OUT = HERE / "data" / "rolling_prediction"


def _users(t0_db):
    c = sqlite3.connect(f"file:{t0_db}?mode=ro", uri=True)
    return [r[0] for r in c.execute("select user_id from users order by user_id")]


def _profile_raw(doc):
    """Turn the public rolling profile contract back into harness prior state."""
    out = {**(doc.get("structured") or {}), **(doc.get("evidence_summary") or {}),
           **(doc.get("presentation") or {})}
    out["data_quality"] = doc.get("data_quality") or {}
    provenance = doc.get("provenance") or {}
    out["agent_sources"] = provenance.get("agent_sources") or {}
    out["evidence_fingerprints"] = provenance.get("evidence_fingerprints") or {}
    out["_meta"] = provenance.get("harness_meta") or {}
    out["version"] = doc.get("profile_version")
    out["as_of"] = doc.get("cutoff_et")
    return out


def _load_t0(base_dir, uid):
    path = Path(base_dir) / f"{uid}.json"
    if not path.exists():
        raise ValueError(f"missing T0 profile for {uid}: {path}")
    return validate_profile(json.loads(path.read_text(encoding="utf-8")))


def _known_holdings(rows, day, *, exact=False):
    usable = [r for r in rows if (r.get("snapshot_date") == day.isoformat()
                                  if exact else
                                  r.get("snapshot_date", "") <= day.isoformat())]
    for row in reversed(usable):
        holdings = row.get("holdings") or {}
        if holdings:
            return holdings, True, row.get("snapshot_date")
        if row.get("stock_total") in (None, 0, 0.0):
            return {}, True, row.get("snapshot_date")
    return {}, False, None


def _coverage(ds, uid, target, execution_coverage_through, target_ds):
    """Boundary books for the target day.

    The two books come from different views on purpose.  `ds` is bounded at the
    cutoff, which is correct for the BEFORE book: it is evidence, and it also
    feeds the market context.  The AFTER book is label-side and sits on the
    target day itself, so it must be read from `target_ds`.  Reading it from the
    cutoff-bounded view silently drops it -- `snapshot_date <= cutoff` can never
    include the target day -- which made every `position_actions` label
    `unknown` and disabled the whole label family.
    """
    daily = ds.get_account_daily(uid)
    before, before_known, before_as_of = _known_holdings(
        daily, target - timedelta(days=1))
    # A stale pre-target book cannot prove the target day's persistent action.
    after, after_known, after_as_of = _known_holdings(
        target_ds.get_account_daily(uid), target, exact=True)
    executions_observed = (execution_coverage_through is not None
                           and target <= execution_coverage_through)
    return {"executions_observed": executions_observed,
            "execution_coverage_basis": ("declared complete through "
                                         f"{execution_coverage_through}"
                                         if execution_coverage_through else
                                         "not declared; no-trade targets remain unknown"),
            "before_book_observed": before_known,
            "after_book_observed": after_known,
            "before_book_as_of": before_as_of, "after_book_as_of": after_as_of,
            "before_book": {"status": "known" if before_known else "unknown",
                            "book_known": before_known, "holdings": before},
            "after_book": {"status": "known" if after_known else "unknown",
                           "book_known": after_known, "holdings": after}}


def _is_current(path):
    """True if `path` holds a sample already written at the current schema.

    Resume must be schema-aware: a sample written by an earlier version is on
    disk and its cutoff is in the immutable store, so a version-blind check
    would skip it forever and leave the dataset mixed.
    """
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(
            "schema_version") == SAMPLE_SCHEMA_VERSION
    except (OSError, json.JSONDecodeError):
        return False


BUSY_EXIT = 75          # EX_TEMPFAIL: another process already owns this user


class _UserLock:
    """Exclusive, advisory, per-user lock held for one user's whole build.

    Two processes building the same user both call `persist_snapshot` for the
    same cutoff, and the store is immutable: the loser raises and that user
    fails after its LLM calls are already paid for.  A file lock turns the
    overlap into a skip.  Every worker is a fresh `python
    build_rolling_dataset.py`, so putting the lock here also coordinates workers
    launched by a supervisor that is running older code.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            return False
        return True

    def __exit__(self, *exc):
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
        return False


def _leakage(profile, recent, market, cutoff):
    violations = []
    if datetime.fromisoformat(profile["cutoff_et"]) > cutoff:
        violations.append("profile cutoff after sample cutoff")
    for row in recent["trades"].get("orders") or []:
        value = row.get("executed_at_et")
        if value and datetime.fromisoformat(value) > cutoff:
            violations.append("recent trade after cutoff")
    market_as_of = market.get("as_of_date")
    if market_as_of and date.fromisoformat(market_as_of[:10]) > cutoff.date():
        violations.append("market bar after cutoff")
    return {"clean": not violations, "violations": violations,
            "cutoff_et": cutoff.isoformat()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t0-db", default=str(HERE / "data" / "bm_t0.db"))
    ap.add_argument("--full-db", default=str(DEFAULT_FULL_DB))
    ap.add_argument("--base-dir", default=str(DEFAULT_OUT / "base_t0"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--market-cache", default=str(DEFAULT_OUT / "market_raw"))
    ap.add_argument("--users")
    ap.add_argument("--users-file", help="ordered user ids, one per line")
    ap.add_argument("--end", default="2026-10-07",
                    help="last calendar date considered; only US market sessions are targets")
    ap.add_argument("--prediction-sessions", type=int, default=40,
                    help="number of US market trading sessions to predict (default 40)")
    ap.add_argument("--execution-coverage-through",
                    help="last date whose BM452 extraction is known complete; without it, no-trade labels are unknown")
    ap.add_argument("--manifest",
                    help="manifest path; default <out>/manifest.json. Give each "
                         "parallel worker its own so they do not clobber one another")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source_users = set(_users(args.t0_db))
    if args.users_file:
        users = [x.strip() for x in Path(args.users_file).read_text().splitlines()
                 if x.strip()]
        unknown = [u for u in users if u not in source_users]
        if unknown:
            raise SystemExit(f"users-file contains ids outside the T0 cohort: {unknown}")
    elif args.users:
        selected = [x.strip() for x in args.users.split(",") if x.strip()]
        users = [u for u in selected if u in source_users]
    else:
        users = sorted(source_users)
        if len(users) != 120:
            raise SystemExit(f"expected the fixed 120-user cohort, found {len(users)}")

    t0 = date.fromisoformat(PROFILE_CUTOFF)
    coverage_through = (date.fromisoformat(args.execution_coverage_through)
                        if args.execution_coverage_through else None)
    post_t0 = trading_days(t0 + timedelta(days=1), date.fromisoformat(args.end))
    if len(post_t0) < 8:
        raise SystemExit("need at least eight post-T0 trading days")
    targets = post_t0[7:7 + args.prediction_sessions]
    if len(targets) < args.prediction_sessions:
        raise SystemExit(f"requested {args.prediction_sessions} prediction sessions but "
                         f"--end only supplies {len(targets)}; extend --end")
    if args.dry_run:
        print(json.dumps({"users": len(users), "T0": PROFILE_CUTOFF,
                          "first_window": [d.isoformat() for d in post_t0[:7]],
                          "first_target": targets[0].isoformat(),
                          "targets": len(targets)}, indent=2))
        return

    out = Path(args.out)
    sample_root = out / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    market_cache = YahooMarketCache(Path(args.market_cache))
    rolling_db = out / "rolling_profiles.db"
    full = LinkBMDataSource(args.full_db)
    manifest = {"schema_version": SAMPLE_SCHEMA_VERSION, "T0": PROFILE_CUTOFF,
                "users": len(users), "targets": len(targets), "samples": 0,
                "failures": [], "busy": [], "leakage_violations": 0}

    for uid in users:
        try:
            # One builder per user: two processes reaching the same cutoff both
            # call `persist_snapshot`, and the store is immutable, so the loser
            # fails the user after its LLM calls are already paid for.
            with _UserLock(out / "locks" / f"{uid}.lock") as owned:
                if not owned:
                    manifest["busy"].append(uid)
                    continue
                prior_doc = _load_t0(args.base_dir, uid)
                prior = _profile_raw(prior_doc)
                user_dir = sample_root / uid
                user_dir.mkdir(parents=True, exist_ok=True)
                for target in targets:
                    cutoff = cutoff_for(target)
                    sample_path = user_dir / f"{target.isoformat()}.json"
                    rolling_store = RollingProfileStore(rolling_db)
                    exact_row = rolling_store.get(uid, cutoff.isoformat())
                    if exact_row and _is_current(sample_path):
                        manifest["samples"] += 1
                        continue
                    ds = full.at(cutoff, include_conversations=False)
                    if exact_row:
                        # Resume after an interruption between the immutable profile write
                        # and the sample write. Never rerun an already-materialised cutoff.
                        profile_doc = validate_profile(json.loads(exact_row["profile_json"]))
                        prior = _profile_raw(profile_doc)
                    else:
                        existing_row = rolling_store.latest_before(uid, cutoff.isoformat())
                        if existing_row:
                            prior_doc = validate_profile(json.loads(existing_row["profile_json"]))
                            prior = _profile_raw(prior_doc)
                        result = refresh(args.full_db, uid, cutoff_et=cutoff.isoformat(),
                                         existing_profile=prior, persist_profile=False)
                        run_meta = {"target_date": target.isoformat(),
                                    "gate": (result.get("update_decision") or {}).get("action"),
                                    "agents": (result.get("execution_plan") or {}).get("agents") or [],
                                    "mode": (result.get("execution_plan") or {}).get("mode")}
                        profile_doc, _ = persist_snapshot(
                            rolling_db, uid, cutoff.isoformat(),
                            result.get("final_profile") or prior, run_meta=run_meta)
                    recent = build_recent_7d(ds, uid, cutoff.date())
                    # Label-side view: extends through the end of the target day.
                    target_ds = full.at(datetime.combine(
                        target, datetime.max.time(), tzinfo=NEW_YORK))
                    coverage = _coverage(ds, uid, target, coverage_through, target_ds)
                    target_trades = [r for r in target_ds.get_trades(uid)
                                     if parse_execution_et(r.get("executed_at")).date() == target]
                    holdings = (coverage.get("before_book") or {}).get("holdings") or {}
                    positions = [{"ticker": ticker,
                                  "market_value": ((row or {}).get("stock_market_value")
                                                   or ((row or {}).get("stock_vol") or 0)
                                                   * ((row or {}).get("last_price") or 0))}
                                 for ticker, row in holdings.items()]
                    # Strictly cutoff-side candidates: target executions are label-only.
                    recent.setdefault("ticker_candidates", {})["current_holdings"] = \
                        sorted(str(ticker) for ticker in holdings)
                    watchlist = recent["behavior"].get("viewed_tickers") or []
                    market = build_market_context(cutoff.date(), positions, watchlist,
                                                  market_cache.load)
                    labels = build_targets(target_trades, target,
                                           coverage.get("before_book"),
                                           coverage.get("after_book"), coverage)
                    public_coverage = {k: v for k, v in coverage.items()
                                       if k not in ("before_book", "after_book")}
                    # The store keeps the full profile for carry-forward; the sample
                    # publishes only the summary layers.
                    published = public_profile(profile_doc)
                    sample = {
                        "schema_version": SAMPLE_SCHEMA_VERSION,
                        "sample_id": f"{uid}_{target.isoformat()}", "user_id": uid,
                        "cutoff_et": cutoff.isoformat(), "prediction_date": target.isoformat(),
                        "profile_snapshot": published, "recent_7d": recent,
                        "market_7d": market, "targets": labels,
                        "observation_coverage": public_coverage,
                        # The deterministic engine's forecast, kept out of the
                        # evidence and archived here as the non-LLM baseline over
                        # the same pre-cutoff inputs, alongside whatever the
                        # profiler LLM wrote so the two can be compared.
                        "baselines": {
                            "rule_engine_forecast": forecast_baseline(profile_doc),
                            "llm_next_action": llm_restatement(profile_doc),
                        },
                        "leakage_audit": _leakage(published, recent, market, cutoff),
                        "provenance": {"full_db": args.full_db,
                                       "market_cache": str(args.market_cache)},
                    }
                    validate_sample(sample)
                    # Build the model-facing view now, not at prediction time: its
                    # tripwires (token budget, forecast fields) then fail the sample
                    # that caused them instead of a whole prediction run later.
                    sample["view_meta"] = build_prompt_view(sample)["view_meta"]
                    sample_path.write_text(
                        json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
                    manifest["samples"] += 1
        except Exception as exc:  # one user cannot hide the rest of the manifest
            manifest["failures"].append({"user_id": uid,
                                         "error": f"{type(exc).__name__}: {exc}"})
    manifest_path = Path(args.manifest) if args.manifest else out / "manifest.json"
    idle = not manifest["samples"] and not manifest["failures"] and manifest["busy"]
    if not idle:
        # A run that only found locked users has nothing to report, and writing
        # anyway would overwrite the manifest of the worker that did the work.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                 + "\n", encoding="utf-8")
    if manifest["failures"]:
        raise SystemExit(f"{len(manifest['failures'])} users failed; see manifest")
    print(f"wrote {manifest['samples']} samples for "
          f"{len(users) - len(manifest['busy'])} users to {out}")
    if manifest["busy"]:
        # Not an error: another builder owns these users. A distinct exit code
        # lets a supervisor requeue them instead of recording a failure.
        print(f"skipped {len(manifest['busy'])} users locked elsewhere: "
              f"{manifest['busy']}")
        raise SystemExit(BUSY_EXIT)


if __name__ == "__main__":
    main()
