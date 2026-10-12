"""Profile lifecycle: freshness, carry-forward and evidence fingerprints.

Two problems meet here.

**P0-4 (staleness).** Harness v2 stops recomputing every dimension on every
refresh. That is only safe if a value we did NOT recompute is identifiable as
such. Without it, "skipped" and "fresh" look identical downstream and an
incremental refresh quietly serves week-old conclusions as current. So a carried
value always travels with `carried_forward`, `age_days` and `stale`.

**Fingerprints.** Freshness cannot be measured by "did new rows arrive". Two
cases break that:

  * the data was REPAIRED rather than extended — this happened for real: an
    invalid `asset_total` on 4.4% of BM453 day-rows produced `invested_ratio`
    above 1.0 for 33 users (peak 205x). After the fix those numbers changed
    while the row count did not;
  * rows were added but changed nothing a given dimension cares about.

So each dimension records a fingerprint of the evidence it actually consumed,
plus the version stamps of the engines and metric blocks that produced it. If
either moves, the dimension is recomputed. Correctness outranks token savings:
when the two disagree, we spend the tokens.
"""
import hashlib
import json
from datetime import datetime, timezone

# How long an unrecomputed dimension may be served before it MUST be refreshed.
# Driven by how fast each concept actually changes, not by convenience:
#   logic      — tied to whatever theme the user is currently researching (fast)
#   risk       — revealed behaviour is a regime, not a daily reading
#   preference — persistent style; movers are event-driven
#   assessment — expertise / lifecycle barely move within a year
DIM_TTL_DAYS = {
    "risk": 30,
    "logic": 7,
    "preference": 30,
    "emotion": 30,
    "assessment": 90,
}

# profile key <-> short dimension name used by POLICY / agent_sources
DIM_KEY = {
    "risk": "risk",
    "logic": "investment_logic",
    "preference": "preferences",
    "emotion": "emotion_and_psychology",
    "assessment": "investor_assessment",
}
KEY_DIM = {v: k for k, v in DIM_KEY.items()}

# Engines / metric blocks whose version invalidates a dimension when bumped.
VERSION_STAMP_KEYS = (
    "position_dynamics", "attention_capital", "trade_metrics",
    "portfolio_metrics", "asset",
)


def _parse(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def age_days(profile, as_of=None):
    """Age of a stored profile in days (None when it has no usable timestamp)."""
    if not profile:
        return None
    ref = _parse(as_of) or datetime.now(timezone.utc)
    for key in ("as_of", "created_at", "updated_at"):
        t = _parse(profile.get(key))
        if t is not None:
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=timezone.utc)
            return round((ref - t).total_seconds() / 86400.0, 2)
    return None


def engine_versions(evidence):
    """Version stamps of everything that fed a dimension's numbers."""
    out = {}
    for key in VERSION_STAMP_KEYS:
        blk = (evidence or {}).get(key) or {}
        v = blk.get("version")
        if v:
            out[key] = v
    return out


def feature_version():
    """Version of the DETERMINISTIC feature layer, from the module constants.

    Read from the modules rather than from a built evidence package, because the
    evidence cache must be able to compute this key BEFORE it has any evidence.
    Bumping any engine invalidates every cache entry.

    EVERY engine that contributes a top-level key to the evidence package must be
    listed. Adding one without bumping this is a silent, partial failure: the cache
    keeps serving packages built by the previous code, so the new key is simply
    absent for users whose data did not otherwise move — and `assemble` guards with
    `if ev.get(key)`, so the missing block is not an error, it is an omission.
    Observed exactly this way: `action_forecast` appeared for 1 of 12 users after
    being wired into `build_evidence`, because only that user's cache entry had
    expired for an unrelated reason.
    """
    from .action_forecast import FORECAST_VERSION
    from .attention_capital import ATTENTION_CAPITAL_VERSION
    from .evidence import ASSET_EVIDENCE_VERSION, POSITION_EVIDENCE_VERSION
    from .portfolio_metrics import METRIC_VERSION as _PM
    from .position_events import POSITION_VERSION
    from .trade_metrics import METRIC_VERSION as _TM
    return "|".join([
        ATTENTION_CAPITAL_VERSION, POSITION_VERSION, _TM, _PM,
        ASSET_EVIDENCE_VERSION, POSITION_EVIDENCE_VERSION, FORECAST_VERSION,
    ])


def evidence_fingerprint(evidence, keys, versions=None):
    """Stable hash of the exact evidence subset a dimension consumed.

    Only the keys that dimension is allowed to see are hashed, so unrelated
    streams growing does not invalidate it. Version stamps are folded in so a
    metric-definition change (a data repair) forces recomputation even when the
    underlying rows are byte-identical.

    `versions` should also carry the PRODUCER identity for the dimension — the
    profiler prompt version and the resolved model. Changing either changes the
    output while leaving the input untouched, so without them a prompt edit or a
    model re-tiering would never be applied to an existing profile.
    """
    subset = {}
    for k in sorted(keys or ()):
        v = (evidence or {}).get(k)
        if v is None:
            continue
        subset[k] = v
    blob = json.dumps(subset, ensure_ascii=False, sort_keys=True, default=str)
    h = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    stamps = json.dumps(versions or {}, sort_keys=True)
    if stamps and stamps != "{}":
        h = h + "-" + hashlib.sha1(stamps.encode("utf-8")).hexdigest()[:8]
    return h


def dim_freshness(existing_profile, as_of=None):
    """Per-dimension freshness of a stored profile.

    Returns {dim: {present, age_days, stale, confidence, ttl_days, fingerprint}}.
    `present` is False when the stored profile has no value for that dimension,
    which is what forces a first-time build.
    """
    prof = existing_profile or {}
    age = age_days(prof, as_of)
    fingerprints = prof.get("evidence_fingerprints") or {}
    out = {}
    for dim, key in DIM_KEY.items():
        val = prof.get(key)
        has = isinstance(val, dict) and bool(val)
        ttl = DIM_TTL_DAYS[dim]
        stale = age is None or age > ttl
        out[dim] = {
            "present": has,
            "age_days": age,
            "ttl_days": ttl,
            "stale": stale or not has,
            "confidence": (val or {}).get("confidence"),
            "fingerprint": fingerprints.get(dim),
        }
    return out


def changed_dimensions(existing_profile, evidence, policy, new_fingerprints):
    """Dimensions whose evidence subset (or engine version) actually moved.

    This is the deterministic half of the delta detector: no LLM, no heuristics
    on row counts.
    """
    old = (existing_profile or {}).get("evidence_fingerprints") or {}
    moved = []
    for dim in DIM_KEY:
        keys = policy.get(dim) or []
        new_fp = new_fingerprints.get(dim)
        if new_fp and old.get(dim) != new_fp:
            moved.append(dim)
    return moved


def carry_forward(dim, existing_profile, as_of=None):
    """Reuse a stored dimension verbatim, tagged as carried.

    Returns None when there is nothing to carry (first build, or the stored
    value is missing) so the caller falls back to the deterministic rule layer
    rather than inventing a value.
    """
    prof = existing_profile or {}
    val = prof.get(DIM_KEY[dim])
    if not isinstance(val, dict) or not val:
        return None
    age = age_days(prof, as_of)
    ttl = DIM_TTL_DAYS[dim]
    carried = dict(val)
    carried["carried_forward"] = True
    carried["carried_from_version"] = prof.get("version")
    carried["age_days"] = age
    carried["stale"] = age is None or age > ttl
    carried["notes"] = ((str(carried.get("notes") or "") +
                         f" | Not recalculated this round; using version {prof.get('version')} ({age} days ago)").strip(" |"))
    return carried


def freshness_block(existing_profile, as_of, plan, reasons):
    """Audit record of what was recomputed vs reused this refresh."""
    fresh = dim_freshness(existing_profile, as_of)
    return {
        "as_of": as_of,
        "profile_age_days": age_days(existing_profile, as_of),
        "recomputed": sorted(plan.get("agents") or []),
        "reused": sorted(d for d in DIM_KEY if d not in (plan.get("agents") or [])),
        "mode": plan.get("mode"),
        "dimensions": {d: {**v, "reason": reasons.get(d)} for d, v in fresh.items()},
    }
