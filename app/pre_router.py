"""Deterministic pre-router: decide the plan WITHOUT spending an LLM call.

Most refreshes do not need the orchestrator. "Nothing profile-relevant changed"
is answerable in Python from the delta and the per-dimension fingerprints, and a
no-change refresh should cost zero tokens rather than one orchestrator call plus
whatever agents it picks.

The division of labour is deliberate:

  * this module decides the plan whenever the answer is determinate, and marks
    whether the LLM is allowed to refine it;
  * the orchestrator LLM is consulted ONLY for the genuinely ambiguous case
    (evidence moved but which dimension matters is unclear);
  * `enforce_plan` still adjudicates whatever comes back, so a wrong LLM
    proposal can never exceed budget or drop a stale dimension.

Note on "no change": a dimension is forced to run when its evidence MOVED, not
merely when rows arrived. Row counts would miss a data repair, which is exactly
how a corrected metric could be left un-applied.
"""
from .lifecycle import DIM_KEY, dim_freshness
from .orchestrator import MODE_BUDGET


def decide(evidence, data_quality, existing_profile, as_of, moved_dims,
           fingerprints):
    """Return a routing decision dict.

    must_run    dimensions that MUST be recomputed (missing or expired values).
                These are never dropped: skipping them would serve a stale value
                while still spending the call.
    candidates  dimensions whose evidence MOVED. They are eligible to run and may
                be ranked/capped; when the answer is determinate and they fit in
                budget they all run.
    mode        fast | standard | deep
    skip_llm    True when the deterministic answer is already correct, so the
                orchestrator call is not made at all.
    """
    existing = existing_profile or {}

    # 1. first build: nothing to carry, everything with data runs (DEEP)
    if not existing:
        dims = [d for d in DIM_KEY if _has_data(evidence, d)]
        return {"must_run": dims, "candidates": [], "mode": "deep",
                "skip_llm": True, "reason": "first_profile:full_build"}

    fresh = dim_freshness(existing, as_of)

    # 2. missing or expired values must be produced regardless of delta
    must = [d for d, s in fresh.items() if not s["present"] or s["stale"]]

    # 3. evidence moved since the stored profile was written
    moved = [d for d in (moved_dims or []) if d not in must]

    # 4. nothing to do: keep every field, spend nothing
    if not must and not moved:
        return {"must_run": [], "candidates": [], "mode": "fast",
                "skip_llm": True, "reason": "no_relevant_delta"}

    # 5. expiry is already a decision, not a question
    if must and not moved:
        mode = "deep" if len(must) >= 3 else "standard"
        return {"must_run": must, "candidates": [], "mode": mode,
                "skip_llm": True, "reason": "stale_dimensions_forced"}

    # 6. moved dims fit in budget: no ranking needed, so no LLM needed
    budget = MODE_BUDGET["standard"] - len(must)
    if not must and len(moved) <= MODE_BUDGET["standard"]:
        return {"must_run": [], "candidates": moved, "mode": "standard",
                "skip_llm": True, "reason": "deterministic_delta"}
    if len(moved) <= max(budget, 0):
        return {"must_run": must, "candidates": moved, "mode": "standard",
                "skip_llm": True, "reason": "deterministic_delta_with_expiry"}

    # 7. ambiguous: more moving dimensions than budget. Let the orchestrator rank
    #    them. It may only choose from `candidates`, and `must_run` is preserved.
    #    When must_run alone already exceeds the standard budget the refresh is a
    #    rebuild in all but name, so label it DEEP.
    mode = "deep" if len(must) >= 3 else "standard"
    return {"must_run": must, "candidates": moved, "mode": mode,
            "skip_llm": False, "reason": "ambiguous_delta"}


def _has_data(evidence, dim):
    """Cheap presence probe mirroring orchestrator._dim_strength > 0."""
    from .orchestrator import _dim_strength
    try:
        return _dim_strength(evidence, dim) > 0
    except Exception:                                                 # noqa: BLE001
        return False
