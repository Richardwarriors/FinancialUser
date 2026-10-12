"""Financial User Profile Harness v2 (LangGraph runtime).

  load_existing -> fetch_snapshot -> validate_data --insufficient--> skip
       -> build_evidence -> delta/pre_router (deterministic, no LLM)
            |-- nothing profile-relevant changed --> carry_forward -> END (0 LLM)
            `-- otherwise -> orchestrator (LLM plan only when ambiguous)
       -> [selected profilers] (parallel; unselected dims carried forward)
       -> merger -> checker -> conflict_resolution (deterministic) -> update_gate
       -> narrate (portrait/KYC reused within TTL) -> write_profile -> END

The orchestrator agent decides WHICH specialists run and at what evidence
depth; permissions, budget, staleness and the update gate remain deterministic
harness controls. The default is NOT "run five agents": a dimension is only
recomputed when its evidence actually moved or its stored value expired.
"""
import contextvars
import json
import os
import re
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph


def _merge_dict(a, b):
    return {**(a or {}), **(b or {})}

from .datasource import get_datasource
from .evidence import build_evidence
from .lifecycle import (DIM_KEY, age_days, carry_forward, changed_dimensions,
                        engine_versions, evidence_fingerprint, feature_version,
                        freshness_block, dim_freshness)
from .pre_router import decide as route_decision
from .snapshot import build_snapshot
from .profilers import (POLICY, PROMPT_VERSION, run_profiler)
from .orchestrator import (ALL_DIMS, agent_view, orchestrate, policy_view,
                           project_evidence)
from .store import SQLiteProfileStore

PIPELINE_VERSION = "harness-v2.0"
_ROLLING_CONTEXT = contextvars.ContextVar("rolling_profile_context", default=None)

# Served-freshness windows for the derived (presentation) layer: the portrait and
# the two trees. Rebuilding these is what v2 made conditional — but their rebuild
# decision is made by app/derived.py against their OWN contract, NOT by the
# dimension gate. Sharing the gate's decision meant a stable user could never
# receive a first-time tree, a version migration or a TTL refresh, because the
# gate answered `no_change` and the derived layer was skipped. Env overrides are
# applied on top of derived.DERIVED_TTL_DAYS.
from .derived import DERIVED_TTL_DAYS, ORDER as DERIVED_ORDER, stale_components

PORTRAIT_TTL_DAYS = int(os.environ.get("PROFILE_PORTRAIT_TTL_DAYS", "7"))
KYC_TTL_DAYS = int(os.environ.get("PROFILE_KYC_TTL_DAYS", "30"))
# the profile tree is derived from the KYC tree + dimensions, so it is reusable
# under the same conditions; 4 summary calls make it worth caching
PROFILE_TREE_TTL_DAYS = int(os.environ.get("PROFILE_PROFILE_TREE_TTL_DAYS", "30"))

DERIVED_TTLS = {"portrait": PORTRAIT_TTL_DAYS, "kyc_tree": KYC_TTL_DAYS,
                "profile_tree": PROFILE_TREE_TTL_DAYS}

# insufficient-data thresholds (deterministic policy)
MIN_EVENTS = 5
MIN_ACCOUNT_DAYS = 5


class FinancialProfileState(TypedDict, total=False):
    job_id: str
    user_id: str
    db_path: str
    as_of: str
    raw_snapshot: dict
    existing_profile: dict
    data_quality: dict
    behavior_intent: dict
    evidence: dict
    evidence_cache: dict
    execution_plan: dict
    fingerprints: dict
    freshness: dict
    risk_candidate: dict
    logic_candidate: dict
    preference_candidate: dict
    emotion_candidate: dict
    assessment_candidate: dict
    agent_sources: Annotated[dict, _merge_dict]
    candidate_profile: dict
    validation: dict
    conflicts: list
    divergence_records: list
    update_decision: dict
    final_profile: dict
    derived_rebuilt: list
    derived_stale_reasons: dict
    model_version: str
    prompt_version: str
    pipeline_version: str


def _audit(db_path, task, user_id, status, detail):
    if (_ROLLING_CONTEXT.get() or {}).get("persist_profile") is False:
        return
    import sqlite3
    c = sqlite3.connect(db_path)
    c.execute("""create table if not exists audit_log(id integer primary key autoincrement,
        task text,user_id text,status text,detail text,created_at text default current_timestamp)""")
    c.execute("insert into audit_log(task,user_id,status,detail) values(?,?,?,?)",
              (task, user_id, status, json.dumps(detail, ensure_ascii=False)))
    c.commit()


# ---------- nodes ----------

def load_existing(state):
    ctx = _ROLLING_CONTEXT.get() or {}
    if ctx.get("existing_profile") is not None:
        return {"existing_profile": ctx["existing_profile"]}
    store = SQLiteProfileStore(state["db_path"])
    row = store.get_latest(state["user_id"])
    existing = json.loads(row["profile_json"]) if row else {}
    return {"existing_profile": existing}


def fetch_snapshot(state):
    ds = get_datasource(os.environ.get("PROFILE_SOURCE_KIND", "link"), state["db_path"])
    cutoff = (_ROLLING_CONTEXT.get() or {}).get("cutoff_et")
    if cutoff:
        ds = ds.at(cutoff, include_conversations=False)
    snap = build_snapshot(ds, state["user_id"], as_of=cutoff)
    return {"raw_snapshot": snap.to_dict(),
            "as_of": snap.as_of,
            "model_version": os.environ.get("LUNA_MODEL", "gpt-5.6-luna"),
            "prompt_version": PROMPT_VERSION,
            "pipeline_version": PIPELINE_VERSION}


def validate_data(state):
    s = state["raw_snapshot"]
    acct = s.get("account_daily") or []
    latest_holdings = (acct[-1].get("holdings") if acct else {}) or {}
    # structured behavior_events intent metrics are computed here so orchestrator, gate
    # and the read-only viewer all see the same numbers
    from .behavior import build_behavior_intent
    intent = build_behavior_intent(s)
    rd = intent.get("research_depth", {}) if intent.get("available") else {}
    rb = intent.get("research_breadth", {}) if intent.get("available") else {}
    dq = {
        "events": len(s["behaviors_6m"]),
        "trades": len(s["trades_6m"]),
        "account_days": len(acct),
        "holdings_latest": len(latest_holdings),
        "conversations": len(s["conversations_1m"]),
        "intent_available": bool(intent.get("available")),
        "unique_tickers": rb.get("unique_tickers"),
        "research_sessions": rd.get("sessions"),
        "research_grade_session_share": rd.get("research_grade_session_share"),
        "conviction_tickers": len(intent.get("conviction_tickers") or []),
    }
    dq["sufficient"] = (dq["events"] > 0 or dq["trades"] > 0
                        or dq["account_days"] > 0 or dq["conversations"] > 0)
    # `richness` is advisory: it says whether the evidence is thick enough for a
    # full-confidence profile, and is reported instead of being used to refuse a
    # user entirely. With the union user universe, a user may first appear as a
    # single event or conversation; the contract is to build what we can and let
    # later runs fill the gaps.
    dq["rich"] = (dq["events"] >= MIN_EVENTS or dq["trades"] > 0
                  or dq["account_days"] >= MIN_ACCOUNT_DAYS)
    dq["streams_present"] = [k for k, n in
                             (("behavior", dq["events"]), ("trade", dq["trades"]),
                              ("account", dq["account_days"]),
                              ("conversation", dq["conversations"])) if n]
    return {"data_quality": dq, "behavior_intent": intent}


def route_after_validation(state):
    return "build_evidence" if state["data_quality"]["sufficient"] else "skip"


def build_evidence_node(state):
    """Deterministic evidence, served from cache when the inputs are unchanged.

    The cache key is a content hash of the raw snapshot plus the engine versions,
    so a data repair invalidates it exactly like new data would. Any cache error
    falls through to recomputation: the cache may never break profile generation.
    """
    from .evidence_cache import CACHE_VERSION, get, put, source_hash
    snap = state["raw_snapshot"]
    src = source_hash(snap)
    fv = f"{CACHE_VERSION}|{feature_version()}"
    rolling_read_only = (_ROLLING_CONTEXT.get() or {}).get("persist_profile") is False
    cached = None if rolling_read_only else get(state["db_path"], state["user_id"], src, fv)
    if cached is not None:
        return {"evidence": cached, "evidence_cache": {"hit": True, "source_hash": src}}
    ev = build_evidence(snap, state.get("behavior_intent"))
    if not rolling_read_only:
        put(state["db_path"], state["user_id"], src, fv, ev)
    return {"evidence": ev, "evidence_cache": {"hit": False, "source_hash": src}}


def _wrap(dim):
    """Profiler node: LLM only when the execution plan selected it.

    A dimension that was NOT selected is carried forward from the stored profile
    when one exists (tagged with its age so nothing downstream mistakes it for a
    fresh value). Only when there is nothing to carry does it fall back to the
    free deterministic rule layer. Either way the profile schema stays complete —
    the orchestrator saves LLM budget, never fields.
    """
    def node(state):
        plan = state.get("execution_plan", {})
        if dim not in plan.get("agents", ALL_DIMS):
            carried = carry_forward(dim, state.get("existing_profile"),
                                    state.get("as_of"))
            if carried is not None:
                return {f"{dim}_candidate": carried,
                        "agent_sources": {dim: "carried_forward"}}
            from .profilers import RULES
            return {f"{dim}_candidate": RULES[dim](policy_view(state["evidence"], dim)),
                    "agent_sources": {dim: "rule:orchestrator_skipped"}}
        view = agent_view(state["evidence"], dim, plan.get("evidence_depth", "standard"))
        # mode comes from the plan: DEEP promotes every dimension to the strong tier
        out, src = run_profiler(dim, view, mode=plan.get("mode"),
                                fallback_evidence=policy_view(state["evidence"], dim))
        return {f"{dim}_candidate": out, "agent_sources": {dim: src}}
    return node


def _evidence_block_names(evidence):
    """Block names a citation may legitimately use as a `block.field` prefix.

    Current blocks come from POLICY plus whatever the evidence actually holds.
    RETIRED ones must be listed explicitly: `stated` was removed from POLICY when
    the person-identifying self-report channel was deleted, so nothing in the live
    configuration mentions it any more — and a guard that only knows the current
    blocks cannot recognise a citation of a removed one, which is precisely the case
    that has to be caught.
    """
    retired = {
        "stated",          # v2.4: users.base_profile_summary (PII) — see PII_REMOVAL.md
    }
    return ({k for keys in POLICY.values() for k in keys}
            | set(evidence or {}) | retired)


def _ungroundable_dims(existing, evidence):
    """Stored dimensions whose cited fields no longer exist in the evidence.

    Carrying a dimension forward is only safe if its `key_signals` still refer to
    real evidence. When the metric layer changes underneath it — this actually
    happened: `asset.avg_stock_value_ratio` and `execution.hold_days_ratio_loser_
    over_winner` were removed as duplicates, and the equity-curve family moved to
    a single owner — a carried value would keep citing fields that are gone, and
    the grounding check would skip it precisely because it is carried.

    Such a dimension is promoted to "moved" so it is rebuilt against the current
    evidence. Self-healing rather than accumulating drift.
    """
    blob = json.dumps(evidence or {}, ensure_ascii=False)
    # Every block name the system knows about, whether or not it is currently
    # present. A citation whose TOP-LEVEL segment is one of these but is ABSENT from
    # the evidence is definitively ungroundable — it cites a channel that no longer
    # exists. This is precise, unlike the token heuristic below: after the `stated`
    # channel was removed, citations reading `stated.stated_background=…` survived
    # because `background` and `stated_vs_revealed` still appear elsewhere in the
    # evidence, so the any-token rule judged them grounded and carried them forever.
    known_blocks = _evidence_block_names(evidence)
    out = []
    for dim, key in DIM_KEY.items():
        val = (existing or {}).get(key)
        if not isinstance(val, dict):
            continue
        for s in val.get("key_signals") or []:
            text = str(s)
            if any(seg in known_blocks and seg not in (evidence or {})
                   for seg in re.findall(r"\b([a-z_]{3,})\.[a-z_]", text)):
                out.append(dim)
                break
            tokens = [t for t in re.findall(r"[A-Za-z_]{4,}", text)
                      if t not in ("true", "false", "null", "None")]
            if tokens and not any(t in blob for t in tokens):
                out.append(dim)
                break
    return out


def profile_router(state):
    """Deterministic pre-router first; orchestrator LLM only if ambiguous.

    Saves the orchestrator call entirely when Python can already answer "what
    changed, and does it justify recomputation".
    """
    ev = state["evidence"]
    dq = state.get("data_quality") or {}
    existing = state.get("existing_profile") or {}
    base_versions = engine_versions(ev)
    # Producer identity is part of a dimension's fingerprint: a prompt edit or a
    # model re-tiering changes the output while leaving the evidence untouched,
    # so without this the change would never reach an existing profile.
    from .profilers import PROMPT_VERSION as _PV, resolve_model
    # NB: the profile-tree version deliberately is NOT part of this fingerprint.
    # The tree is rebuilt by `narrate`, which compares PROFILE_TREE_VERSION itself,
    # and it is assembled from the dimension outputs rather than the other way
    # round. Including it here would make every presentation change re-run all 5
    # dimensions for all 157 users -- a full-budget rebuild to change a heading.
    base_versions = {**base_versions, "profiler_prompt": _PV}
    # the standard-tier model identifies the producer. DEEP only matters on a first
    # build or a stale dimension, both of which are forced to run anyway, so the
    # mode is deliberately not part of the fingerprint.
    fingerprints = {
        d: evidence_fingerprint(ev, POLICY.get(d, []),
                                {**base_versions, "model": resolve_model(d)})
        for d in ALL_DIMS}
    moved = changed_dimensions(existing, ev, POLICY, fingerprints)
    # a stored dimension citing fields that no longer exist must be rebuilt, not
    # carried — see _ungroundable_dims
    broken = [d for d in _ungroundable_dims(existing, ev) if d not in moved]
    moved = sorted(set(moved) | set(broken))
    decision = route_decision(ev, dq, existing, state.get("as_of"), moved, fingerprints)
    must_run, candidates = decision["must_run"], decision["candidates"]

    fresh = dim_freshness(existing, state.get("as_of"))
    profile_state = {d: {k: v for k, v in fresh[d].items()
                         if k in ("age_days", "stale", "confidence", "present")}
                     for d in ALL_DIMS}
    plan = orchestrate(ev, dq, force=must_run, candidates=candidates,
                       profile_state=profile_state,
                       skip_llm=decision["skip_llm"])
    plan["mode"] = decision["mode"] if decision["skip_llm"] else plan.get("mode", decision["mode"])
    plan["router"] = {"decision": decision["reason"],
                      "skip_llm": decision["skip_llm"],
                      "moved_dimensions": moved,
                      "ungroundable_dimensions": broken,
                      "must_run": must_run,
                      "candidates": candidates}
    _audit(state["db_path"], "harness_orchestrate", state["user_id"], "planned", plan)
    return {"execution_plan": plan, "fingerprints": fingerprints,
            "freshness": freshness_block(existing, state.get("as_of"), plan,
                                         {d: decision["reason"] for d in ALL_DIMS})}


def merger(state):
    r = state.get("risk_candidate", {})
    l = state.get("logic_candidate", {})
    p = state.get("preference_candidate", {})
    e = state.get("emotion_candidate", {})
    a = state.get("assessment_candidate", {})
    conflicts = []
    # deterministic pre-merge consistency probes
    if p.get("trading_style") == "long_term_investor" and r.get("revealed_risk_behavior") in ("aggressive", "very_aggressive"):
        conflicts.append({"field": "style_vs_risk", "detail": "long_term label with aggressive revealed risk", "severity": "low"})
    if (r.get("revealed_risk_behavior") not in (None, "unknown")) and p.get("stated_vs_revealed") == "divergent":
        conflicts.append({"field": "stated_vs_revealed", "detail": p.get("notes") or "divergence flagged", "severity": "medium"})
    if a.get("experience_level") == "beginner" and r.get("revealed_risk_behavior") in ("aggressive", "very_aggressive"):
        conflicts.append({"field": "experience_vs_risk", "detail": "beginner self-report with aggressive revealed behavior", "severity": "medium"})
    if e.get("chasing_momentum") == "yes" and p.get("trading_style") == "long_term_investor":
        conflicts.append({"field": "emotion_vs_style", "detail": "momentum chasing against long-term style claim", "severity": "low"})
    candidate = {
        "user_id": state["user_id"],
        "profile_status": "inferred",
        "as_of": state.get("as_of"),
        "risk": r, "investment_logic": l, "preferences": p,
        "emotion_and_psychology": e, "investor_assessment": a,
        "limitations": ["Behavioral profile only; not financial advice or suitability.",
                        "Stated risk tolerance requires verified KYC data.",
                        "conversations conversation titles carry no timestamps: recency and full dialogue context are not verified.",
                        "Account metrics come from account_snapshots daily labels; deposits/withdrawals can distort equity returns."],
    }
    return {"candidate_profile": candidate, "conflicts": conflicts}


DIMS = {"risk": "risk", "investment_logic": "logic", "preferences": "preference",
        "emotion_and_psychology": "emotion", "investor_assessment": "assessment"}


def checker(state):
    """Deterministic validation ladder (no LLM).

    Grounding is checked against the SAME projected evidence the model was given,
    not the full package. Validating against everything was weaker than it
    looked: a signal naming a field that `project_evidence` had trimmed away
    would pass, even though the model could not have seen it.
    """
    errs = []
    c = state.get("candidate_profile", {})
    plan = state.get("execution_plan", {})
    depth = plan.get("evidence_depth", "standard")
    ev = state.get("evidence", {})
    # `DIMS` maps PROFILE keys -> short names; POLICY and the projection are keyed
    # by the SHORT name. Passing the profile key to agent_view returned an empty
    # view, which made every signal of that dimension look ungrounded.
    for profile_key, dim in DIMS.items():
        d = c.get(profile_key) or {}
        # ground against the EXACT view the agent was given (permission filter +
        # projection). Checking against a superset let a signal citing a block the
        # dimension may not read pass validation.
        view = agent_view(ev, dim, depth)
        blob = json.dumps(view, ensure_ascii=False)
        full_blob = json.dumps(policy_view(ev, dim), ensure_ascii=False)
        carried = bool(d.get("carried_forward"))
        if not isinstance(d.get("confidence"), (int, float)) or not 0 <= float(d["confidence"]) <= 1:
            errs.append(f"{profile_key}.confidence out of range")
        if not d.get("evidence_strength"):
            errs.append(f"{profile_key}.evidence_strength missing")
        # A carried value was grounded against the evidence of the version it
        # came from; re-grounding it against today's blob would fail purely
        # because the data moved on. Freshness is the check that applies to it.
        if carried:
            if d.get("age_days") is None:
                errs.append(f"{profile_key} carried forward without age_days")
            continue
        for s in d.get("key_signals", []):
            tokens = [t for t in re.findall(r"[A-Za-z_]{4,}", str(s)) if t not in ("true", "false", "null", "None")]
            if tokens and not any(t in blob for t in tokens):
                # distinguish "trimmed by projection" from "never existed": the
                # former is a real problem (the model cited something it could not
                # have seen), the latter means the model invented a field
                why = "in the permitted evidence but trimmed by the projection" \
                    if any(t in full_blob for t in tokens) else "not in any permitted evidence"
                errs.append(f"{profile_key} key_signal ungrounded ({why}): {s[:60]}")

    # The forecast is the one field that looks FORWARD, so it is the easiest place
    # for a model to write something confident and unfalsifiable. It is anchored to
    # the deterministic engine: the model may refine the engine's candidates, never
    # add one the engine did not score, and never claim more confidence than the
    # engine's own (which already drops when the evidence is thin).
    na = ((c.get("investment_logic") or {}).get("next_action")) or {}
    if na:
        fc = ev.get("action_forecast") or {}
        primary = na.get("primary_action")
        if not fc.get("available"):
            if primary not in (None, "unknown"):
                errs.append("investment_logic.next_action set although the forecast "
                            "engine reported available=false")
        else:
            scored = {x.get("action") for x in (fc.get("candidates") or [])}
            if primary != "unknown" and primary not in scored:
                errs.append(f"investment_logic.next_action primary_action "
                            f"{primary!r} was never scored by the engine "
                            f"(scored: {sorted(scored)})")
            ceiling = round(float(fc.get("confidence") or 0) + 0.1, 4)
            try:
                got = float(na.get("confidence") or 0)
            except (TypeError, ValueError):
                got = 1.0
            if got > ceiling:
                errs.append(f"investment_logic.next_action confidence {got} exceeds "
                            f"the engine ceiling {ceiling}")
            allowed = set()
            for x in (fc.get("candidates") or []):
                allowed |= {t.get("ticker") for t in (x.get("targets") or [])}
            if not na.get("falsifier"):
                errs.append("investment_logic.next_action has no falsifier "
                            "(an unfalsifiable forecast cannot be checked)")
            bad = [t for t in (na.get("targets") or []) if t and t not in allowed]
            if bad:
                errs.append(f"investment_logic.next_action targets not in the "
                            f"engine's candidate set: {bad}")
    return {"validation": {"ok": not errs, "errors": errs}}


def conflict_resolution(state):
    """Record divergence between dimensions. No LLM, and NO field rewriting.

    The conflicts this system actually produces are cross-FIELD tensions, not two
    agents claiming the same field: measured on the 12 demo users, all 6 were
    `style_vs_risk` (2), `experience_vs_risk` (2) and `emotion_vs_style` (2).
    Both sides are true observations of the same user — someone who describes
    themselves as a long-term investor while behaving aggressively is exactly the
    finding worth keeping. There is nothing to "resolve".

    The previous implementation called an LLM which, on action=resolve, directly
    edited `preferences.stated_vs_revealed`, `preferences.notes` and
    `investor_assessment.notes`. That is the "merger silently changes the meaning"
    failure mode, just relocated into the conflict stage. Now the divergence is
    recorded and both values are left exactly as the agents wrote them.
    """
    conflicts = state.get("conflicts", [])
    if not conflicts:
        return {"update_decision": {"action": "approve"}}

    divergences = []
    for f in conflicts:
        field = f.get("field")
        pair = {
            "style_vs_risk": ("preferences.trading_style", "risk.revealed_risk_behavior"),
            "experience_vs_risk": ("investor_assessment.experience_level", "risk.revealed_risk_behavior"),
            "emotion_vs_style": ("emotion_and_psychology.chasing_momentum", "preferences.trading_style"),
            "stated_vs_revealed": ("preferences.stated_vs_revealed", "preferences.trading_style"),
        }.get(field, (None, None))
        divergences.append({
            "field": field,
            "between": [p for p in pair if p],
            "detail": f.get("detail"),
            "severity": f.get("severity"),
            "resolution": "retained_both",
            "note": "The two dimensions describe different aspects of the same user. Both are observation results and should not be merged or rewritten",
        })
    return {"update_decision": {"action": "approve_with_flags", "flags": conflicts,
                               "divergences": divergences},
            "divergence_records": divergences}


def update_gate(state):
    """Pure deterministic policy: strong-enough evidence only."""
    v, dec = state.get("validation", {}), state.get("update_decision", {})
    plan = state.get("execution_plan") or {}
    cands = [state.get("risk_candidate", {}), state.get("logic_candidate", {}),
             state.get("preference_candidate", {}), state.get("emotion_candidate", {}),
             state.get("assessment_candidate", {})]
    # Zero-delta refresh: nothing was recomputed, so writing a new version would
    # create version churn with byte-identical dimensions and no new information.
    #
    # This is only safe for a genuine no-delta decision. If the router found moved
    # dimensions, skipping the write would leave the fingerprints unpersisted and
    # the same dimensions would appear "moved" on every subsequent refresh — a
    # silent non-convergence loop. `enforce_plan` guarantees at least one agent
    # runs in that case, so `agents == []` here really does mean "no delta".
    router = plan.get("router") or {}
    if (not plan.get("agents") and state.get("existing_profile")
            and not router.get("moved_dimensions")):
        return _gate(state, "no_change",
                     f"router: {router.get('decision', 'no relevant delta')}")
    if not v.get("ok"):
        return _gate(state, "keep_old", f"validation failed: {v.get('errors')}")
    if dec.get("action") == "reject":
        return _gate(state, "keep_old", "conflict rejected by review")
    good = [x for x in cands if x.get("evidence_strength") in ("medium", "high")
            and float(x.get("confidence") or 0) >= 0.4]
    if not good:
        # A user we have never profiled still gets a profile when any stream has
        # data. Refusing here meant someone appearing in only one financial data feature
        # never got a profile at all, and they can never be enriched later
        # because there is nothing to enrich. It is written as `partial` so the
        # low confidence is explicit rather than hidden, and it is never allowed
        # to REPLACE an existing profile (that path still holds old below).
        dq = state.get("data_quality") or {}
        if not state.get("existing_profile") and dq.get("streams_present"):
            final = {**state["candidate_profile"],
                     "profile_status": "partial",
                     "execution_plan": state.get("execution_plan", {}),
                     "agent_sources": state.get("agent_sources", {}),
                     "data_quality": dq,
                     "conflicts": state.get("conflicts", []),
                     "gate": {"decision": "approve_partial",
                              "reason": "first profile from thin evidence",
                              "decided_at": datetime.now(timezone.utc).isoformat()}}
            return {"final_profile": final, "update_decision": {**dec, "action": "approve"}}
        return _gate(state, "keep_old", "all candidates low-evidence")
    final = {**state["candidate_profile"],
             "execution_plan": state.get("execution_plan", {}),
             "agent_sources": state.get("agent_sources", {}),
             "data_quality": state.get("data_quality", {}),
             "conflicts": state.get("conflicts", []),
             "gate": {"decision": "approve", "decided_at": datetime.now(timezone.utc).isoformat()}}
    return {"final_profile": final, "update_decision": {**dec, "action": "approve"}}


def _gate(state, action, reason):
    old = state.get("existing_profile") or {}
    return {"update_decision": {"action": action, "reason": reason},
            "final_profile": old or None}


def narrate(state):
    """Attach the human-readable layer to a profile.

    Two independent questions meet here, and keeping them separate is the point
    of this node:

      * the gate has already decided whether the DIMENSION values changed;
      * this node decides whether the DERIVED layer (portrait / KYC tree /
        profile tree) is still built by the current code from the current
        upstream (see app/derived.py).

    Collapsing them into one decision is what made a new presentation layer
    effectively unreachable: the gate said `no_change`, this node returned before
    touching the derived layer, and `write_profile` discarded the result. So a
    first-time tree, a version migration, or a TTL refresh could never reach a
    user whose dimensions were stable — which is the steady state, not an edge
    case.

    Failures stay contained: narration must never break profile generation.
    """
    final = state.get("final_profile")
    dec = state.get("update_decision", {})
    action = dec.get("action")
    if not final or action == "skip":
        return {}
    existing = state.get("existing_profile") or {}
    plan = state.get("execution_plan") or {}
    recomputed = list(plan.get("agents") or ())
    validation = state.get("validation") or {}
    age = age_days(existing, state.get("as_of"))

    # Never build prose on top of data that failed validation: the numbers it
    # would summarise are the suspect ones, so the stored profile is the safer
    # answer. (The gate can return `no_change` before it checks validation.)
    #
    # The metric-key set is computed BEFORE the early return below, because a
    # user with none of the five blocks (no trades, no positions) must not report
    # `metrics` stale forever — see current_metric_version's docstring. The filter
    # is `applicable_metric_keys`, not mere presence: a block stored as
    # `{available: False}` is a statement that the metric does not apply, and it
    # carries no version, so counting it would churn on every refresh.
    from .derived import METRIC_BLOCKS as _METRIC_BLOCKS
    from .derived import applicable_metric_keys
    _ev = state.get("evidence") or {}
    metric_keys = tuple(k for k in _METRIC_BLOCKS
                        if k in applicable_metric_keys(_ev)
                        or k in applicable_metric_keys(existing))
    todo = ({name: why for name, why in
             stale_components(existing, state.get("as_of"), recomputed,
                              ttl_days=DERIVED_TTLS, metric_keys=metric_keys).items()}
            if validation.get("ok") else {})
    if action != "approve" and not todo:
        # Nothing recomputed and nothing stale: the 0-token path that makes a
        # stable fleet cheap. Keep the audit trail the old code wrote here.
        if action == "no_change":
            _audit(state["db_path"], "harness_refresh", state["user_id"], "no_change",
                   {"reason": dec.get("reason"),
                    "fingerprints": state.get("fingerprints") or {}})
        return {}

    ev = state.get("evidence") or {}
    out = dict(final if action == "approve" else (existing or final))
    reused, rebuilt = {}, []

    # --- portrait (1 LLM call) ------------------------------------------------
    # A summary of the five dimensions, so it is stale exactly when a dimension
    # was recomputed or its prompt version moved.
    if "portrait" in todo:
        from .narrative import build_portrait
        try:
            portrait = build_portrait(final, state.get("data_quality"))
            portrait["reused"] = False
        except Exception as e:                                    # noqa: BLE001
            portrait = {"enabled": False, "reason": f"{type(e).__name__}",
                        "headline": "", "summary": "", "highlights": [], "caveats": [],
                        "reused": False}
        out["portrait"] = portrait
        rebuilt.append("portrait")
    elif isinstance(out.get("portrait"), dict):
        portrait = {**out["portrait"], "reused": True}
        if age is not None:
            portrait["age_days"] = age
        out["portrait"] = portrait
        reused["portrait"] = todo.get("portrait") or f"current, age {age}d"

    # --- KYC tree (3 LLM calls) ----------------------------------------------
    # 60 mostly-stable leaves. Held here because profile_tree embeds its content,
    # so the tree below must see whether this was rebuilt.
    kyc_tree = out.get("kyc_tree") if isinstance(out.get("kyc_tree"), dict) else None
    if "kyc_tree" in todo:
        try:
            from .kyc_tree import build_kyc_tree
            fresh = build_kyc_tree(state["raw_snapshot"], state["evidence"],
                                   state.get("behavior_intent"))
            if fresh:
                fresh["reused"] = False
                kyc_tree = fresh
                out["kyc_tree"] = fresh
                rebuilt.append("kyc_tree")
        except Exception as e:                                    # noqa: BLE001
            # On failure keep whatever was already stored rather than dropping the
            # tree: a stale tree beats no tree, and the audit row records why.
            _audit(state["db_path"], "harness_kyc", state["user_id"], "failed",
                   {"error": f"{type(e).__name__}: {e}"})
    elif kyc_tree:
        kyc_tree = {**kyc_tree, "reused": True}
        if age is not None:
            kyc_tree["age_days"] = age
        out["kyc_tree"] = kyc_tree
        reused["kyc_tree"] = todo.get("kyc_tree") or f"current, age {age}d"

    # --- profile tree (4 LLM calls for the L1/L2 summaries) -------------------
    # L3 leaves are assembled deterministically (no LLM); only the prose costs
    # tokens, so the summaries are the part that is worth serving from cache.
    if "profile_tree" in todo:
        try:
            from .profile_tree import apply_summaries, build_profile_tree
            from .profile_tree_agents import summarize
            # The first argument is the PROFILE (the dimension outputs), not the
            # snapshot. Passing `raw_snapshot` here meant every `("dim", ...)` L2
            # source resolved against an object that has no dimension keys, so it
            # silently yielded nothing — which is why `risk_preference`,
            # `investment_logic` and `investor_assessment` all shipped with ZERO L2
            # in the stored tree while the KYC-sourced ones (which read other
            # context keys) worked. A wrong argument that degrades to "empty" is the
            # expensive kind: no error, no gap, just less profile than intended.
            tree = build_profile_tree(out, kyc_tree,
                                      ev.get("attention_capital"),
                                      state.get("behavior_intent"),
                                      state["raw_snapshot"],
                                      evidence=ev)
            payload, tsrc = summarize(tree)
            tree = apply_summaries(tree, payload)
            tree["summary_source"] = tsrc
            tree["reused"] = False
            out["profile_tree"] = tree
            rebuilt.append("profile_tree")
        except Exception as e:                                    # noqa: BLE001
            _audit(state["db_path"], "harness_profile_tree", state["user_id"], "failed",
                   {"error": f"{type(e).__name__}: {e}"})
    elif isinstance(out.get("profile_tree"), dict):
        tree = {**out["profile_tree"], "reused": True}
        if age is not None:
            tree["age_days"] = age
        out["profile_tree"] = tree
        reused["profile_tree"] = todo.get("profile_tree") or f"current, age {age}d"

    # computed metric blocks: deterministic, no LLM, so they are always attached
    # (they are large, which is why the profilers only ever saw the digest).
    # `asset` is the single owner of the equity-curve family (return / drawdown /
    # volatility), so the viewer reads those from here rather than from a second
    # copy inside portfolio_metrics.
    for key in ("trade_metrics", "portfolio_metrics", "asset",
                "position_dynamics", "attention_capital", "action_forecast"):
        if ev.get(key):
            out[key] = ev[key]
    # ...but "always attached" was not the same as "always WRITTEN". On a
    # `no_change` decision this node used to bail out before reaching here, and
    # `write_profile` then discarded the result, so a metric-definition fix
    # (POSITION_VERSION / ATTENTION_CAPITAL_VERSION) never reached a user whose
    # dimensions were quiet. `metrics` is the derived-layer name for this group of
    # five blocks: when their composite stamp differs from what is stored, the
    # blocks above ARE a rebuild and must be persisted.
    if "metrics" in todo:
        rebuilt.append("metrics")
    # A carried dimension keeps the fingerprint of the run that PRODUCED it.
    #
    # `profile_router` computes a fingerprint for every dimension, and persisting
    # all of them meant a carried dimension adopted the CURRENT one — claiming it
    # had consumed evidence it never saw. The consequence is severe and silent: any
    # change to that dimension's inputs (a removed evidence channel, a prompt edit,
    # a model re-tiering) is absorbed, `changed_dimensions` sees "no delta" forever,
    # and the stored text can never be corrected. This is how person-identifying
    # text survived a migration that had already removed its source: the dimensions
    # citing it were carried, so they reported themselves current.
    #
    # Keeping the producing run's fingerprint makes the next refresh see the delta
    # and run the dimension. Convergence then costs a few passes rather than one,
    # which is the intended price of the per-refresh budget cap.
    new_fp = dict(state.get("fingerprints") or {})
    old_fp = (existing.get("evidence_fingerprints") or {})
    ran = set(recomputed)
    for _dim, _key in DIM_KEY.items():
        if _dim not in ran and old_fp.get(_dim):
            new_fp[_dim] = old_fp[_dim]
    out["evidence_fingerprints"] = new_fp
    out["freshness"] = state.get("freshness") or {}
    if reused:
        out["reused_components"] = reused
    if state.get("divergence_records"):
        out["divergences"] = state["divergence_records"]
    # `derived_rebuilt` / `derived_stale_reasons` describe THIS RUN, so they are
    # returned as state keys and never written into the profile document. Persisting
    # them made a quiet run read back the previous run's rebuild list, i.e. the
    # stored profile claimed a rebuild that did not happen. It is also why
    # `write_profile` reads them from state rather than from the profile.
    return {"final_profile": out, "derived_rebuilt": rebuilt,
            "derived_stale_reasons": todo}


def write_profile(state):
    final = state.get("final_profile")
    dec = state.get("update_decision", {})
    if (_ROLLING_CONTEXT.get() or {}).get("persist_profile") is False:
        return {"final_profile": final or state.get("existing_profile") or {}}
    action = dec.get("action")
    rebuilt = list(state.get("derived_rebuilt") or [])
    store = SQLiteProfileStore(state["db_path"])
    if action in ("keep_old", "no_change"):
        # The dimension values did not change, so the only thing that can
        # legitimately be written is a refreshed derived layer — and it goes to
        # the EXISTING version. Minting a new version here would fill the history
        # with rows whose business content is byte-identical, which makes "what
        # changed, and when" unanswerable. That is the one question the version
        # history exists to answer.
        if not final or not rebuilt:
            _audit(state["db_path"], "harness_refresh", state["user_id"], "skipped",
                   {"reason": dec.get("reason"), "decision": action,
                    "derived_rebuilt": rebuilt})
            return {"final_profile": state.get("existing_profile") or {}}
        version = store.update_latest(state["user_id"], final)
        if version is None:
            # No stored row to refresh (should not happen: the gate only returns
            # a non-approve action when a profile already exists).
            _audit(state["db_path"], "harness_refresh", state["user_id"], "skipped",
                   {"reason": "derived rebuild but no stored version",
                    "decision": action, "derived_rebuilt": rebuilt})
            return {"final_profile": state.get("existing_profile") or {}}
        _audit(state["db_path"], "harness_refresh", state["user_id"], "derived_refreshed",
               {"version": version, "derived_rebuilt": rebuilt, "decision": action,
                "reason": dec.get("reason")})
        return {"final_profile": {**final, "version": version}}
    if not final:
        _audit(state["db_path"], "harness_refresh", state["user_id"], "skipped",
               {"reason": dec.get("reason"), "decision": action})
        return {"final_profile": state.get("existing_profile") or {}}
    version = store.save(state["user_id"], final,
                         [{"dimension": k, "signals": v.get("key_signals", [])} for k, v in
                          [("risk", state.get("risk_candidate", {})),
                           ("investment_logic", state.get("logic_candidate", {})),
                           ("preferences", state.get("preference_candidate", {})),
                           ("emotion_and_psychology", state.get("emotion_candidate", {})),
                           ("investor_assessment", state.get("assessment_candidate", {}))]],
                         {"model_version": state.get("model_version"),
                          "prompt_version": state.get("prompt_version"),
                          "pipeline_version": state.get("pipeline_version")})
    _audit(state["db_path"], "harness_refresh", state["user_id"], "success",
           {"version": version, "gate": dec, "derived_rebuilt": rebuilt})
    return {"final_profile": {**final, "version": version}}


def skip_node(state):
    _audit(state["db_path"], "harness_refresh", state["user_id"], "insufficient_data",
           state.get("data_quality", {}))
    return {"update_decision": {"action": "skip", "reason": "insufficient data"},
            "final_profile": state.get("existing_profile") or {}}


# ---------- graph ----------

def build_graph():
    g = StateGraph(FinancialProfileState)
    g.add_node("load_existing", load_existing)
    g.add_node("fetch_snapshot", fetch_snapshot)
    g.add_node("validate_data", validate_data)
    g.add_node("build_evidence", build_evidence_node)
    g.add_node("profile_router", profile_router)
    g.add_node("risk_agent", _wrap("risk"))
    g.add_node("logic_agent", _wrap("logic"))
    g.add_node("preference_agent", _wrap("preference"))
    g.add_node("emotion_agent", _wrap("emotion"))
    g.add_node("assessment_agent", _wrap("assessment"))
    g.add_node("merger", merger)
    g.add_node("checker", checker)
    g.add_node("conflict_resolution", conflict_resolution)
    g.add_node("update_gate", update_gate)
    g.add_node("narrate", narrate)
    g.add_node("write_profile", write_profile)
    g.add_node("skip", skip_node)

    g.set_entry_point("load_existing")
    g.add_edge("load_existing", "fetch_snapshot")
    g.add_edge("fetch_snapshot", "validate_data")
    g.add_conditional_edges("validate_data", route_after_validation,
                            {"build_evidence": "build_evidence", "skip": "skip"})
    g.add_edge("build_evidence", "profile_router")
    g.add_edge("profile_router", "risk_agent")
    g.add_edge("profile_router", "logic_agent")
    g.add_edge("profile_router", "preference_agent")
    g.add_edge("profile_router", "emotion_agent")
    g.add_edge("profile_router", "assessment_agent")
    g.add_edge("risk_agent", "merger")
    g.add_edge("logic_agent", "merger")
    g.add_edge("preference_agent", "merger")
    g.add_edge("emotion_agent", "merger")
    g.add_edge("assessment_agent", "merger")
    g.add_edge("merger", "checker")
    g.add_edge("checker", "conflict_resolution")
    g.add_edge("conflict_resolution", "update_gate")
    g.add_edge("update_gate", "narrate")
    g.add_edge("narrate", "write_profile")
    g.add_edge("write_profile", END)
    g.add_edge("skip", END)
    return g.compile()


def refresh(db_path, user_id, *, cutoff_et=None, existing_profile=None,
            persist_profile=True):
    """Run one full refresh and return the graph output.

    Rolling callers pass an aware ``cutoff_et``, the prior rolling snapshot and
    ``persist_profile=False``.  This keeps the production store untouched while the
    rolling store records one immutable profile per cutoff.

    Token accounting is started here so every LLM call in the refresh is
    attributed to this user. Production refreshes write usage to the audit log;
    rolling backtests return usage in memory and leave the source database read-only.
    """
    import time as _time
    from .token_meter import report, start
    job_id = f"{user_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start()
    t0 = _time.time()
    initial = {"job_id": job_id, "user_id": user_id, "db_path": str(db_path)}
    token = _ROLLING_CONTEXT.set({"cutoff_et": (str(cutoff_et)
                                                 if cutoff_et is not None else None),
                                  "existing_profile": existing_profile,
                                  "persist_profile": persist_profile})
    try:
        out = build_graph().invoke(initial)
    finally:
        _ROLLING_CONTEXT.reset(token)
    elapsed = round(_time.time() - t0, 3)
    usage = report() or {}
    plan = (out.get("execution_plan") or {})
    usage["elapsed_s"] = elapsed
    usage["mode"] = plan.get("mode")
    usage["agents"] = plan.get("agents") or []
    usage["gate"] = (out.get("update_decision") or {}).get("action")
    if persist_profile:
        _audit(str(db_path), "harness_usage", str(user_id), "measured", usage)
    else:
        out["rolling_usage"] = usage
    return out
