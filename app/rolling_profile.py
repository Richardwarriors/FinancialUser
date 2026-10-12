"""Build/export the stable point-in-time profile contract used by rolling samples."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from .english_output import assert_english, englishize
from .forecast_scrub import scrub_prose
from .rolling_schema import (BANNED_PUBLISHED_TRAIT_FIELDS,
                             BANNED_TRAIT_KEY_SUBSTRINGS,
                             PROFILE_SCHEMA_VERSION, PROFILE_DIMENSIONS,
                             PUBLIC_PROFILE_SCHEMA_VERSION, validate_profile,
                             validate_public_profile)
from .rolling_store import RollingProfileStore

DIMENSIONS = ("risk", "investment_logic", "preferences",
              "emotion_and_psychology", "investor_assessment")
COMPUTED = ("trade_metrics", "portfolio_metrics", "asset",
            "position_dynamics", "attention_capital", "action_forecast")


def _base_profile(profile):
    portrait = profile.get("portrait") or {}
    tree = profile.get("profile_tree") or {}
    summaries = {}
    for key, node in (tree.get("l1") or {}).items():
        if node.get("summary"):
            summaries[key] = node["summary"]
    summary = portrait.get("summary")
    if not summary:
        narratives = [str((profile.get(k) or {}).get("narrative") or "").strip()
                      for k in DIMENSIONS]
        summary = " ".join(x for x in narratives if x)
    return {
        "headline": portrait.get("headline"),
        "summary": summary or "No supported profile conclusion at this cutoff.",
        "dimensions": summaries,
        "highlights": portrait.get("highlights") or [],
        "caveats": portrait.get("caveats") or profile.get("limitations") or [],
    }


def export_profile(user_id, profile, cutoff_et, version=None, parent_version=None,
                   run_meta=None):
    cutoff = datetime.fromisoformat(str(cutoff_et).replace("Z", "+00:00"))
    structured = {k: deepcopy(profile[k]) for k in DIMENSIONS if profile.get(k)}
    computed = {k: deepcopy(profile[k]) for k in COMPUTED if profile.get(k)}
    doc = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "user_id": str(user_id),
        "profile_version": int(version if version is not None
                               else profile.get("version") or 0),
        "parent_version": parent_version,
        "cutoff_et": cutoff.isoformat(),
        "profile_date": cutoff.date().isoformat(),
        "base_profile": _base_profile(profile),
        "structured": structured,
        "evidence_summary": computed,
        "presentation": {k: deepcopy(profile[k]) for k in
                         ("portrait", "profile_tree", "kyc_tree") if profile.get(k)},
        "data_quality": deepcopy(profile.get("data_quality") or {}),
        "provenance": {
            "agent_sources": deepcopy(profile.get("agent_sources") or {}),
            "evidence_fingerprints": deepcopy(profile.get("evidence_fingerprints") or {}),
            "harness_meta": deepcopy(profile.get("_meta") or {}),
            "rolling_run": deepcopy(run_meta or {}),
        },
    }
    doc = englishize(doc)
    assert_english(doc)
    return validate_profile(doc)


def persist_snapshot(db_path, user_id, cutoff_et, profile, run_meta=None):
    store = RollingProfileStore(db_path)
    parent = store.latest_before(user_id, cutoff_et)
    next_version = int(parent["version"]) + 1 if parent else 1
    doc = export_profile(user_id, profile, cutoff_et, next_version,
                         int(parent["version"]) if parent else None, run_meta)
    version, created = store.save(user_id, cutoff_et, doc, run_meta or {})
    doc["profile_version"] = version
    return doc, created


def _banned_trait_keys(node: dict):
    return [k for k in node if k in BANNED_PUBLISHED_TRAIT_FIELDS
            or any(s in k.lower() for s in BANNED_TRAIT_KEY_SUBSTRINGS)]


def _strip_forecast_fields(node):
    """Recursively remove rule-forecast fields from a public projection."""
    if isinstance(node, dict):
        return {key: _strip_forecast_fields(value)
                for key, value in node.items()
                if key not in _banned_trait_keys(node)}
    if isinstance(node, list):
        return [_strip_forecast_fields(value) for value in node]
    return deepcopy(node)


def forecast_baseline(doc: dict) -> dict:
    """The non-LLM next-action forecast, archived at sample level.

    Read from `evidence_summary.action_forecast`, the deterministic engine's own
    output -- NOT from `structured.investment_logic.next_action`, which is only
    the profiler LLM's transcription of it.  The transcription is 98% faithful,
    but a baseline labelled "rule engine" must not be LLM-mediated: across the
    80-user cohort 21 versions left it empty and 6 named a different action, and
    every one of those would have entered the paper as an engine failure.  The
    engine itself produced a ranking for 1750 of 1760 versions.

    Archived at sample level, never inside the profile a predictor reads.
    """
    fc = (doc.get("evidence_summary") or {}).get("action_forecast") or {}
    cands = fc.get("candidates") or []
    top = cands[0] if cands else {}
    return {
        "available": bool(fc.get("available")) and bool(top),
        "source": " ".join(x for x in ("app/action_forecast.py",
                                       fc.get("version")) if x),
        "primary_action": top.get("action"),
        "score": top.get("score"),
        # The engine's own confidence, which already falls when the behavioural
        # record is thin -- so a low number here is a fact about the evidence.
        "confidence": fc.get("confidence"),
        "margin": fc.get("margin"),
        "runner_up": cands[1].get("action") if len(cands) > 1 else None,
        "ranking": [{"action": c.get("action"), "score": c.get("score")}
                    for c in cands],
        "targets": [t.get("ticker") for t in (top.get("targets") or [])
                    if t.get("ticker")],
        "falsifier": top.get("falsifier"),
        "horizon_days": fc.get("horizon_days"),
        "observed_actions": fc.get("observed_actions"),
    }


def llm_restatement(doc: dict) -> dict:
    """The forecast fields the profiler LLM wrote, which `public_profile` strips.

    Kept next to `forecast_baseline` rather than thrown away: it is what earlier
    builds reported as the rule engine, so archiving both is what makes the
    difference between them measurable instead of a claim.
    """
    out = {}
    for dim, node in (doc.get("structured") or {}).items():
        if not isinstance(node, dict):
            continue
        for key in _banned_trait_keys(node):
            out[f"{dim}.{key}"] = deepcopy(node[key])
    return out


def public_profile(doc: dict) -> dict:
    """Project a stored profile down to what the dataset publishes.

    The internal profile keeps `evidence_summary` and `presentation` because the
    harness feeds the previous version back in to decide which dimensions moved
    and to carry unchanged ones forward.  Those two layers are ~93% of the
    document and are pure audit surface, so they are not published: a sample
    carries the headline summary plus one summary per dimension.
    """
    base = deepcopy(doc.get("base_profile") or {})
    dims = dict(base.get("dimensions") or {})
    structured = doc.get("structured") or {}
    # Guarantee a summary for every dimension: fall back to the dimension's own
    # narrative so a published profile never ships a hole.
    alias = {"risk_preference": "risk", "app_behavior_analysis": "preferences"}
    for dim in PROFILE_DIMENSIONS:
        if (dims.get(dim) or "").strip():
            continue
        node = structured.get(alias.get(dim, dim)) or {}
        text = str(node.get("narrative") or node.get("summary") or "").strip()
        dims[dim] = text or "No supported conclusion for this dimension at this cutoff."
    base["dimensions"] = dims
    # Remove the rule engine's own next-action forecast: it is stated in the label
    # vocabulary, so publishing it inside the evidence would let a predictor copy
    # the answer. `forecast_baseline` archives it at sample level instead.
    base = _strip_forecast_fields(base)
    structured = _strip_forecast_fields(structured)
    # Removing forecast FIELDS is not enough: the profiler receives the same
    # forecast as evidence and can restate it in reader-facing prose. Remove only
    # sentences tied to that mechanism; retain observed historical actions.
    base = scrub_prose(base, "base_profile", prose=False)
    structured = scrub_prose(structured, "structured", prose=False)
    out = {
        "schema_version": PUBLIC_PROFILE_SCHEMA_VERSION,
        "user_id": doc.get("user_id"),
        "profile_version": doc.get("profile_version"),
        "parent_version": doc.get("parent_version"),
        "cutoff_et": doc.get("cutoff_et"),
        "profile_date": doc.get("profile_date"),
        "base_profile": base,
        "structured": deepcopy(structured),
        "data_quality": deepcopy(doc.get("data_quality") or {}),
        "provenance": {"rolling_run": deepcopy(
            (doc.get("provenance") or {}).get("rolling_run") or {})},
    }
    return validate_public_profile(out)
