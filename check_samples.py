#!/usr/bin/env python3
"""Acceptance check for the rolling-prediction dataset.

Run this before treating the build as finished.  It answers the questions the
paper has to answer about its own data, and it fails loudly rather than printing
a reassuring summary over a broken corpus:

    completeness   every cohort user has every target, at the current schema
    English        no CJK, no "English explanation unavailable" placeholder, and
                   no prose the English boundary emptied instead of translating
    contract       one summary plus a summary for each of the five dimensions,
                   and neither audit layer published
    leakage        every sample's audit is clean, and the model-facing view
                   builds -- which re-checks the token budget and refuses any
                   rule-engine forecast field or forecast-derived prose
    labels         distribution of both label families, and how much of each is
                   unobserved (an `unknown` is not a `hold`)
    market         how many samples carry a usable market block
    T0 profile     the published T0 base profile exists for every cohort user and
                   holds the same contract as a sample's `profile_snapshot`

Usage
-----
    ./.venv/bin/python check_samples.py
    ./.venv/bin/python check_samples.py --out data/rolling_prediction/v4
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from app.english_output import contains_cjk                          # noqa: E402
from app.rolling_config import PROFILE_CUTOFF                          # noqa: E402
from app.rolling_prompt import build_prompt_view                     # noqa: E402
from app.rolling_schema import (PROFILE_DIMENSIONS,                  # noqa: E402
                                PUBLIC_PROFILE_SCHEMA_VERSION,
                                SAMPLE_SCHEMA_VERSION,
                                public_profile_contamination,
                                validate_public_profile, validate_sample)
from app.rolling_time import trading_days                            # noqa: E402

PLACEHOLDER = "English explanation unavailable"
PROSE_KEYS = frozenset({"display", "narrative", "summary", "note", "reason",
                        "caveat", "headline", "text", "value", "label",
                        "description"})


def _strings(node, key=""):
    """(owning key, string) for every string anywhere in `node`."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, str(k))
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v, key)
    elif isinstance(node, str):
        yield key, node


def gutted(node) -> list[str]:
    """Prose the English boundary emptied instead of translating.

    `englishize` deletes the Chinese run and keeps whatever machine fragment is
    left, so a Chinese sentence built around numbers comes out as
    `'366 , 40%, 7 ; 91%, 9%, 0%'`.  That carries no CJK and is not the
    placeholder, so neither of the other two English checks can see it: the only
    tell is that the words are gone.
    """
    out = []
    for key, text in _strings(node):
        if key not in PROSE_KEYS or len(text) < 25:
            continue
        if re.fullmatch(r"[\d\-T:+.]+", text):        # a timestamp, not prose
            continue
        if len(re.findall(r"[A-Za-z]{3,}", text)) < 3:
            out.append(f"{key}={text[:60]!r}")
    return out


def check_published_t0(root: Path, users: list[str], problems: list[str]) -> dict:
    """The T0 base profile as the dataset publishes it.

    `base_t0/` is the builder's working seed and has to keep the audit layers,
    because the harness feeds them back as prior state.  The deliverable is the
    `public_profile` projection of it, so it is held to the same contract as a
    sample's `profile_snapshot` and checked here rather than trusted.
    """
    published = root / "base_t0_published"
    present, missing = [], []
    for uid in users:
        path = published / f"{uid}.json"
        if not path.exists():
            missing.append(uid)
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        try:
            validate_public_profile(doc)
        except Exception as exc:                                      # noqa: BLE001
            problems.append(f"published T0 {uid}: {type(exc).__name__}: {exc}")
            continue
        flat = json.dumps(doc, ensure_ascii=False)
        if contains_cjk(flat):
            problems.append(f"published T0 {uid}: contains CJK")
        if PLACEHOLDER in flat:
            problems.append(f"published T0 {uid}: contains {PLACEHOLDER!r}")
        for hit in gutted(doc):
            problems.append(f"published T0 {uid}: prose was emptied: {hit}")
        base = doc.get("base_profile") or {}
        if not (base.get("summary") or "").strip():
            problems.append(f"published T0 {uid}: no profile summary")
        for dim in PROFILE_DIMENSIONS:
            if not ((base.get("dimensions") or {}).get(dim) or "").strip():
                problems.append(f"published T0 {uid}: no summary for {dim}")
        present.append(uid)
    return {"present": len(present), "missing": missing}


def expected_targets(end: str, prediction_sessions: int = 40) -> list[str]:
    """Exactly N US market sessions; weekends and exchange holidays never appear."""
    t0 = date.fromisoformat(PROFILE_CUTOFF)
    post = trading_days(t0 + timedelta(days=1), date.fromisoformat(end))
    targets = post[7:7 + prediction_sessions]
    if len(targets) < prediction_sessions:
        raise ValueError(f"--end supplies {len(targets)} of {prediction_sessions} "
                         "requested prediction sessions")
    return [d.isoformat() for d in targets]


def dimension_sources(db: Path, problems: list[str]) -> dict:
    """How each stored dimension was written, across every profile version.

    `run_profiler` does not re-raise an LLM failure: it substitutes the
    deterministic rule narrative and tags the source `rule:llm_error:<type>`.
    That degrades quietly, so a rate-limited run would produce a full dataset of
    rule-written text and no error anywhere. Counting the tags is the only way
    to see it.
    """
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    counts = Counter()
    for (raw,) in con.execute("select profile_json from rolling_profile_versions"):
        doc = json.loads(raw)
        for source in ((doc.get("provenance") or {})
                       .get("agent_sources") or {}).values():
            counts[str(source)] += 1
    degraded = {k: v for k, v in counts.items() if k.startswith("rule:llm_error")}
    if degraded:
        problems.append(f"{sum(degraded.values())} dimensions fell back to the "
                        f"rule engine after an LLM error: {degraded}")
    return dict(counts.most_common())


def check_sample(sample: dict, problems: list[str]) -> dict:
    """-> per-sample facts; appends any contract breach to `problems`."""
    sid = sample.get("sample_id")
    validate_sample(sample)
    published = sample["profile_snapshot"]
    if published.get("schema_version") != PUBLIC_PROFILE_SCHEMA_VERSION:
        problems.append(f"{sid}: profile_snapshot is not the published contract")
    validate_public_profile(published)
    for hit in public_profile_contamination(published):
        problems.append(f"{sid}: rule-forecast semantic contamination: {hit}")
    flat = json.dumps(sample, ensure_ascii=False)
    if contains_cjk(flat):
        problems.append(f"{sid}: contains CJK")
    if PLACEHOLDER in flat:
        problems.append(f"{sid}: contains {PLACEHOLDER!r}")
    for hit in gutted(sample):
        problems.append(f"{sid}: prose was emptied by the English boundary: {hit}")
    if not (sample.get("leakage_audit") or {}).get("clean"):
        problems.append(f"{sid}: leakage audit not clean")
    base = published.get("base_profile") or {}
    if not (base.get("summary") or "").strip():
        problems.append(f"{sid}: no profile summary")
    dims = base.get("dimensions") or {}
    for dim in PROFILE_DIMENSIONS:
        if not (dims.get(dim) or "").strip():
            problems.append(f"{sid}: dimension summary missing for {dim}")
    # Building the view is itself a check: it refuses a forecast field and
    # enforces the token budget.
    try:
        view = build_prompt_view(sample)
    except ValueError as exc:
        problems.append(f"{sid}: prompt view refused: {exc}")
        view = None
    targets = sample.get("targets") or {}
    trade = targets.get("trade_decision") or {}
    net = targets.get("net_trade_action") or {}
    pos = targets.get("position_actions") or {}
    market = sample.get("market_7d") or {}
    regime = market.get("regime") or {}
    return {
        "activity": (trade.get("trade_activity") if trade.get("is_observed")
                     else "unobserved"),
        "sides": (trade.get("trade_sides") if trade.get("is_observed")
                  else "unobserved"),
        "consistency": trade.get("execution_book_consistency") or "unknown",
        "net": net.get("action") if net.get("is_observed") else "unobserved",
        "position": pos.get("primary_action") if pos.get("is_observed")
        else "unobserved",
        "has_baseline": bool((sample.get("baselines") or {})
                             .get("rule_engine_forecast")),
        "market_usable": any((v or {}).get("status") == "available"
                             for v in regime.values()),
        "view_tokens": view["view_meta"]["approx_tokens"] if view else None,
        "sample_tokens": len(json.dumps(sample, separators=(",", ":"),
                                        ensure_ascii=False)) // 4,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "data" / "rolling_prediction"))
    ap.add_argument("--users-file", default=str(
        HERE / "data" / "users.txt"))
    ap.add_argument("--end", default="2026-10-07")
    ap.add_argument("--prediction-sessions", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.out) / "samples"
    users = [u.strip() for u in Path(args.users_file).read_text().split() if u.strip()]
    targets = expected_targets(args.end, args.prediction_sessions)
    problems: list[str] = []
    activity_dist, sides_dist, consistency_dist = Counter(), Counter(), Counter()
    net_dist, pos_dist = Counter(), Counter()
    view_tokens, sample_tokens = [], []
    complete, partial, missing_users = [], {}, []
    stale = with_baseline = market_ok = 0

    for uid in users:
        d = root / uid
        if not d.is_dir():
            missing_users.append(uid)
            continue
        have = set()
        for day in targets:
            path = d / f"{day}.json"
            if not path.exists():
                continue
            sample = json.loads(path.read_text(encoding="utf-8"))
            if sample.get("schema_version") != SAMPLE_SCHEMA_VERSION:
                stale += 1
                continue
            facts = check_sample(sample, problems)
            have.add(day)
            activity_dist[facts["activity"]] += 1
            sides_dist[facts["sides"]] += 1
            consistency_dist[facts["consistency"]] += 1
            net_dist[facts["net"]] += 1
            pos_dist[facts["position"]] += 1
            with_baseline += facts["has_baseline"]
            market_ok += facts["market_usable"]
            if facts["view_tokens"]:
                view_tokens.append(facts["view_tokens"])
            sample_tokens.append(facts["sample_tokens"])
        if len(have) == len(targets):
            complete.append(uid)
        else:
            partial[uid] = f"{len(have)}/{len(targets)}"

    n = sum(net_dist.values())
    published_t0 = check_published_t0(Path(args.out), users, problems)
    report = {
        "schema": SAMPLE_SCHEMA_VERSION,
        "cohort_users": len(users),
        "users_complete": len(complete),
        "users_partial": partial,
        "users_missing": missing_users,
        "targets_per_user": len(targets),
        "samples_expected": len(users) * len(targets),
        "samples_at_current_schema": n,
        "samples_at_older_schema": stale,
        "problems": problems[:40],
        "n_problems": len(problems),
        "trade_activity": dict(activity_dist),
        "trade_sides": dict(sides_dist),
        "execution_book_consistency": dict(consistency_dist),
        "net_trade_action_legacy": dict(net_dist),
        "position_action_legacy": dict(pos_dist),
        "unobserved_share": {
            "net_trade_action": round(net_dist["unobserved"] / n, 4) if n else None,
            "position_action": round(pos_dist["unobserved"] / n, 4) if n else None,
        },
        "published_t0_profiles": published_t0,
        "dimension_sources": dimension_sources(
            Path(args.out) / "rolling_profiles.db", problems),
        "samples_with_rule_engine_baseline": with_baseline,
        "samples_with_usable_market": market_ok,
        "prompt_view_tokens": {"max": max(view_tokens), "mean":
                               round(sum(view_tokens) / len(view_tokens))}
        if view_tokens else None,
        "archived_sample_tokens": {"max": max(sample_tokens), "mean":
                                   round(sum(sample_tokens) / len(sample_tokens))}
        if sample_tokens else None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    ready = (not problems and not missing_users and not partial
             and not published_t0["missing"]
             and n == len(users) * len(targets))
    print("\nACCEPTED" if ready else "\nNOT READY")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
