"""Orchestrator Agent: dynamic selection & coordination of profiler agents.

Contract:
  - The orchestrator PROPOSES (which agents run, evidence depth, why).
  - The Harness DISPOSES: deterministic enforcement owns permissions, budget
    truncation, core-coverage, and (downstream) validation + update gate.
    A proposal that oversteps is corrected in code, never trusted.

This keeps the LLM as a coordinator, not a permission authority.
"""
import json
import os
from typing import Literal

from pydantic import BaseModel, Field

ORCHESTRATOR_PROMPT_VERSION = "orchestrator-v1"

ALL_DIMS = ["risk", "logic", "preference", "emotion", "assessment"]
# Dimensions whose absence broke a gate semantic. They stay PRIORITISED, but are
# no longer FORCED to run: forcing them capped every refresh at >=3 LLM calls no
# matter how little had changed, which is the single biggest cost driver. A
# dimension that is not rerun is carried forward with staleness metadata
# (see lifecycle.carry_forward) so the profile schema stays complete.
CORE_DIMS = ["risk", "logic", "preference"]
PRIORITY = ["logic", "risk", "preference", "emotion", "assessment"]  # drop-from-tail order

# Execution modes. Budget is a property of the MODE, not a global constant.
MODE_BUDGET = {
    "fast": 1,        # at most one dimension, only if the delta justifies it
    "standard": 2,
    "deep": 5,
}
MAX_AGENT_CALLS = int(os.environ.get("BM_MAX_AGENT_CALLS", "2"))


class Plan(BaseModel):
    agents: list[str] = Field(default_factory=list)
    evidence_depth: Literal["lite", "standard"] = "standard"
    mode: Literal["fast", "standard", "deep"] = "standard"
    rationale: str = ""


def _dim_strength(ev, dim):
    """Deterministic data backing per dimension: how much real evidence exists."""
    b, t = ev.get("behavior", {}), ev.get("trade", {})
    p, a = ev.get("position", {}), ev.get("asset", {})
    cv, s = ev.get("conversation", {}), ev.get("stated", {})
    bi = ev.get("behavior_intent", {})
    intent = 1 if bi.get("available") else 0
    # structured BM433 research evidence is worth more than raw event volume
    deep = 1 if (bi.get("research_depth", {}) or {}).get("research_grade_session_share", 0) \
        and bi["research_depth"]["research_grade_session_share"] >= 0.3 else 0
    backing = {
        "risk": (2 if t.get("available") else 0) + (1 if p.get("available") else 0)
                + (1 if a.get("available") else 0),
        "logic": (2 if (t.get("available") and b.get("available")) else 0)
                 + (1 if b.get("available") else 0) + (1 if cv.get("available") else 0)
                 + intent + deep,
        "preference": (2 if t.get("available") else 0) + (1 if b.get("available") else 0)
                      + (1 if p.get("available") else 0) + (1 if cv.get("available") else 0)
                      + intent,
        "emotion": (1 if b.get("available") else 0) + (1 if t.get("available") else 0)
                   + (1 if p.get("available") else 0) + (1 if cv.get("available") else 0)
                   + (1 if s.get("available") else 0) + intent,
        "assessment": (2 if s.get("available") else 0) + (1 if b.get("available") else 0)
                      + (1 if p.get("available") else 0) + (1 if cv.get("available") else 0)
                      + intent + deep,
    }
    return backing.get(dim, 0)


def _default_plan(ev, dq, force=None, candidates=None):
    """Rule fallback: run everything the data can back; lite when behaviour-heavy.

    `force` must run; `candidates` are eligible and are taken in PRIORITY order
    up to the standard budget. `force is None` means "no routing information",
    which is the first-build / DEEP path.
    """
    if force is None and candidates is None:
        agents = [d for d in ALL_DIMS if _dim_strength(ev, d) > 0]
        mode = "deep"
    else:
        must = [d for d in (force or ()) if d in ALL_DIMS]
        eligible = [d for d in (candidates or ()) if d in ALL_DIMS and d not in must]
        room = max(MODE_BUDGET["standard"] - len(must), 0)
        eligible.sort(key=PRIORITY.index)
        agents = must + eligible[:room]
        mode = "fast" if not agents else "standard"
    agents = sorted(set(a for a in agents if _dim_strength(ev, a) > 0 or (force and a in force)),
                    key=PRIORITY.index)
    heavy = ev.get("behavior", {}).get("event_count", 0) or 0
    depth = "lite" if heavy > 800 and dq.get("trades", 0) == 0 else "standard"
    return Plan(agents=agents, evidence_depth=depth, mode=mode,
                rationale="deterministic default plan (rule fallback)")


def enforce_plan(proposed: Plan, ev: dict, dq: dict, force=None,
                 candidates=None) -> dict:
    """Deterministic adjudication of any proposal (LLM or rule).

    Rules enforced here, in order:
      1. whitelist          — unknown agents are dropped
      2. eligible set       — an agent may only run if it is eligible (a candidate
                              whose evidence moved or was stale) when routing
                              information exists, so an LLM cannot inflate scope
      3. must-run           — stale/absent dimensions are ADDED and never dropped
      4. budget             — capped by MODE_BUDGET, must-run is exempt
      5. coverage           — never run an agent with wholly absent evidence
    """
    agents = [a for a in proposed.agents if a in ALL_DIMS]          # 1. whitelist
    if force is not None or candidates is not None:
        eligible = set(force or ()) | set(candidates or ())         # 2. eligible
        agents = [a for a in agents if a in eligible]
    forced = sorted(set(force or ()) & set(ALL_DIMS))
    agents = sorted(set(agents) | set(forced), key=PRIORITY.index)  # 3. must-run
    budget = MODE_BUDGET.get(proposed.mode, MAX_AGENT_CALLS)
    if proposed.mode != "deep":
        budget = min(budget, MAX_AGENT_CALLS)
    dropped = []
    while len(agents) > budget:                                     # 4. budget cap
        victim = next((a for a in reversed(agents)
                       if a not in CORE_DIMS and a not in forced), None)
        if victim is None:
            victim = next((a for a in reversed(agents) if a not in forced), None)
        if victim is None:
            break                                                   # all must-run
        agents.remove(victim)
        dropped.append(victim)
    # 5. coverage guard
    agents = [a for a in agents if _dim_strength(ev, a) > 0 or a in forced]
    # 6. progress guarantee: if evidence moved, SOMETHING must run.
    #
    # Without this the system does not converge. A proposal that selects nothing
    # while candidates exist leads to update_gate returning `no_change`, which
    # writes no version, which means the fingerprints are never persisted — so the
    # same dimensions look "moved" on the next refresh, forever. Taking the
    # highest-priority candidate costs one call and lets the state advance.
    if not agents and candidates:
        eligible = sorted((c for c in candidates if _dim_strength(ev, c) > 0),
                          key=PRIORITY.index)
        if eligible:
            agents = [eligible[0]]
            dropped.append("progress_guarantee")
    depth = proposed.evidence_depth if proposed.evidence_depth in ("lite", "standard") else "standard"
    return {"agents": sorted(agents, key=PRIORITY.index), "evidence_depth": depth,
            "mode": proposed.mode,
            "rationale": proposed.rationale[:500],
            "must_run": forced,
            "eligible": sorted(set(candidates or ()), key=PRIORITY.index),
            "dropped_by_budget": dropped,
            "policy": {"allowed": ALL_DIMS, "core": CORE_DIMS,
                       "mode_budget": budget, "max_calls": MAX_AGENT_CALLS}}


def project_evidence(ev: dict, depth: str, dim: str = None) -> dict:
    """Evidence view for a dimension: per-dimension whitelist, then depth trim.

    Two reductions, in order:
      1. `dim` applies the per-dimension whitelist (app/evidence_projection.py),
         which removes whole sub-objects a dimension does not reason over. This is
         where the real savings are: measured 101.7 KB -> 59.3 KB across the five
         dimensions (~26k -> ~15k tokens per user).
      2. `depth` still truncates long lists for the `lite` tier.

    `dim=None` returns the full package, which is what the viewer trace wants.
    """
    if dim is not None:
        from .evidence_projection import project
        ev = project(ev, dim)[0]
    if depth != "lite":
        return ev
    view = json.loads(json.dumps(ev, ensure_ascii=False))
    for stream in view.values():
        if not isinstance(stream, dict):
            continue
        for k, v in stream.items():
            if isinstance(v, list) and len(v) > 5:
                stream[k] = v[:5]
    return view


def agent_view(ev: dict, dim: str, depth: str = "standard") -> dict:
    """Exactly what an agent is allowed to see: POLICY first, then projection.

    Single source of truth for the agent's view, used by BOTH the profiler call
    and the grounding check. They used to differ: the check ran against the
    projection without the permission filter, i.e. a superset of what the model
    saw, so a signal citing a block the dimension is not permitted to read could
    pass validation.
    """
    from .profilers import POLICY
    permitted = [k for k in POLICY.get(dim, []) if (ev.get(k) or {}).get("available")]
    subset = {k: ev[k] for k in permitted}
    return project_evidence(subset, depth, dim)


def policy_view(ev: dict, dim: str) -> dict:
    """Policy-filtered evidence WITHOUT the token projection.

    For the deterministic rule layer: it is not token-constrained, but it must
    still respect the information boundary.

    Keys are present even when a block is `available: False`, because the rules
    access them directly (`ev["trade"]`) and branch on `available` themselves.
    Filtering unavailable blocks out here raised KeyError on exactly the users who
    need the rule path most — the ones with no trades at all.
    """
    from .profilers import POLICY
    return {k: ev[k] for k in POLICY.get(dim, []) if k in ev}


def propose_plan(ev: dict, dq: dict, force=None, candidates=None,
                 profile_state=None, skip_llm=False) -> tuple[Plan, str]:
    """LLM proposal with rule fallback.

    The LLM sees summary stats only, never raw rows. When `skip_llm` is set the
    deterministic router already decided the plan, so the call is not made at
    all — this is what makes a no-change refresh cost zero tokens.

    `profile_state` (per-dimension freshness + delta) is preferable input to the
    raw evidence scalars: the decision "which fields have enough new evidence to
    justify recomputation" is a different question from "what is this user's
    profile", and it is far cheaper to answer.
    """
    if skip_llm:
        return _default_plan(ev, dq, force=force, candidates=candidates), "rule:pre_router"
    from .profilers import _client
    client = _client()
    if client is None:
        return _default_plan(ev, dq, force=force, candidates=candidates), "rule"
    from .profilers import resolve_model
    model = resolve_model("logic")      # orchestration is a reasoning task: strong tier
    summary = {d: _dim_strength(ev, d) for d in ALL_DIMS}
    # The orchestrator's question is "which fields have enough NEW evidence to
    # justify recomputation", not "what is this user's profile". Answering the
    # first needs freshness + delta + a coarse strength score; shipping the full
    # evidence scalars (~2k tokens) mainly gives the model room to be distracted
    # by content it is not being asked to judge. profile_state is now the primary
    # input; `evidence_scalars` is sent only when no profile exists yet (nothing
    # to compare against, so the first build cannot be described by a delta).
    payload = {"data_quality": dq, "dim_strength": summary}
    if profile_state:
        payload["profile_state"] = profile_state
        payload["task"] = "rank_eligible_dimensions"
    else:
        compact = {k: {kk: vv for kk, vv in v.items()
                       if not isinstance(vv, list)}
                   for k, v in ev.items() if isinstance(v, dict)}
        payload["evidence_scalars"] = compact
        payload["task"] = "first_build_selection"
    if force is not None:
        payload["must_run"] = sorted(force)
    if candidates is not None:
        payload["eligible"] = sorted(candidates)
    try:
        # include_raw so the provider's token usage is visible; the parsed model is
        # still available under ["parsed"]
        raw_run = client.with_structured_output(Plan, include_raw=True)
        res = raw_run.invoke([
            ("system",
             "You are the profile orchestrator. The goal is not to \"generate a complete profile,\" but to answer: which dimensions have sufficient new evidence to warrant recalculation. Select only from eligible; must_run must be retained and must not be removed. The number of selected dimensions must not exceed the budget for the selected mode. Most incremental refreshes should run only 0-2 dimensions, prioritizing dimensions most relevant to the new evidence. mode values: fast (no material change, agents may be empty), standard (1-2 dimensions), deep (first-time profiling/systemic change). When activity volume is huge and there are no transactions, select lite to save token; select standard when evidence is sufficient. Do not fabricate evidence; give brief reasons."),
            ("user", json.dumps(payload, ensure_ascii=False)),
        ])
        from .token_meter import record
        record("orchestrator", model, getattr(res.get("raw"), "usage_metadata", None))
        out = res.get("parsed")
        if out is None:
            return _default_plan(ev, dq, force=force, candidates=candidates), "rule"
        # keep the declared mode consistent with what was actually proposed, so
        # the budget applied downstream matches the intent
        n = len(set(out.agents or ()) | set(force or ()))
        out.mode = "fast" if n == 0 else "deep" if n >= 4 else "standard"
        return out, "llm"
    except Exception:
        return _default_plan(ev, dq, force=force, candidates=candidates), "rule"


def orchestrate(ev: dict, dq: dict, force=None, candidates=None,
                profile_state=None, skip_llm=False) -> dict:
    proposed, source = propose_plan(ev, dq, force=force, candidates=candidates,
                                    profile_state=profile_state, skip_llm=skip_llm)
    plan = enforce_plan(proposed, ev, dq, force=force, candidates=candidates)
    plan["source"] = source
    plan["prompt_version"] = ORCHESTRATOR_PROMPT_VERSION
    return plan
