"""Profile Tree v2: 10 fixed L1 -> L2 -> L3, ending in readable text.

Replaces the KYC feature tree as the *presentation* structure. The difference is
the shape of the answer, not the facts:

  KYC tree (7 L1)      organised by KYC CATEGORY (goals / risk / portfolio / ...)
  Profile tree (10 L1) organised by WHO THE USER IS and WHAT WE CAN DO FOR THEM
                       (lifecycle / background / goals / constraints / risk /
                        emotion / investment logic / app behaviour /
                        personalisation / assessment)

`portfolio_state` is deliberately NOT an L1 here. Position and trading facts are
an *outcome* of the investor rather than a category of them, so in the tree they
sit under `investor_assessment`, and in the viewer they are carried as charts.

Design decisions that matter
----------------------------
* **L1 is fixed; L2 is curated; L3 is derived.** The 10 L1 keys never move. L2 is
  chosen from the existing KYC label tree and re-parented, so nothing is lost and
  no new facts are invented.
* **L3 is assembled deterministically, not generated.** Every L3 leaf is an
  already-computed fact (60 KYC leaves, the 4 B-class signals, the 5 dimension
  outputs, the computed metric blocks). Re-deriving them with an LLM would cost
  tokens and add a way to be wrong. Only the L1/L2 SUMMARIES are model-written,
  because prose is the one thing that genuinely needs a model.
* **Gaps stay gaps.** A leaf with no data keeps `status="gap"` and carries its
  `data_needed` list into the new tree. Re-parenting must not launder an unknown
  into a fact.
* **Nothing here is allowed to fail the profile.** Summaries are optional; if the
  model is unavailable the tree is still complete and usable.
"""
from datetime import datetime, timezone
import re

from .evidence_glossary import annotate as evidence_annotate
from .evidence_glossary import de_repr
from .pii import find_pii, scrub as pii_scrub

# v2: the L1 contract changed from 11 keys to the 10 fixed keys below. The bump
# matters because the version participates in the dimension fingerprints --
# without it, existing profiles would keep serving a tree built under the old
# shape, and the frontend label map would silently miss an L1.
PROFILE_TREE_VERSION = "profile-tree-v10"


def tree_version_stamp():
    """Version of what the tree CONTAINS: its structure AND its prose.

    The prose comes from `profile_tree_agents`, which carries its own prompt
    version, and changing that prompt changes the tree's content. Keying freshness
    on the structure version alone reproduced the very defect this layer exists to
    fix — the change is written but never reaches a stored tree, because nothing
    declares it stale.

    Imported lazily on purpose: this module is deterministic and must stay free of
    any dependency on the LLM-facing module at import time.
    """
    from .profile_tree_agents import PROFILE_TREE_PROMPT_VERSION
    return f"{PROFILE_TREE_VERSION}+{PROFILE_TREE_PROMPT_VERSION}"

# status vocabulary shared with the KYC tree so the two are comparable
FILLED, PROXY, GAP = "filled", "proxy", "gap"

# ---------------------------------------------------------------------------
# L1 definitions: fixed keys, fixed order.
# Each L2 names ONE source, so a fact lives in exactly one place:
#   ("kyc", <l1>, <l2>)   re-parent an existing KYC L2 (all of its leaves)
#   ("dim", <dimension>)  a dimension output from the 5 profilers
#   ("bs",  <signal>)     a B-class behaviour signal
#   ("blk", <block>, <path>)  a computed metric block (dotted path)
#
# The ("stated",) source was REMOVED: it mounted `users.base_profile_summary` — a
# demographic self-description carrying 国籍 / 年龄 / 语言, and once a locality — as
# `自述原文` under two labels. Person data has no analytical role in an investment
# profile; see app/pii.py for the boundary guard that now covers every source.
# ---------------------------------------------------------------------------
L1_DEFS = {
    "user_life_cycle": {
        "label": "User lifecycle",
        "hint": "Life and wealth stage, occupation, age, and language background. Used to determine suitable service methods.",
        "l2": {
            "lifecycle_stage": ("Wealth lifecycle stage", ("kyc", "jurisdiction_and_lifecycle", "lifecycle")),
            "career": ("Occupation and Employment", ("kyc", "jurisdiction_and_lifecycle", "career")),
            "assessed_stage": ("Assessed stage", ("dim", "investor_assessment_lifecycle")),
        },
    },
    "background": {
        "label": "Background",
        "hint": "Jurisdiction, tax status, income and assets, and user-reported background.",
        "l2": {
            "jurisdiction": ("Jurisdiction and tax", ("kyc", "jurisdiction_and_lifecycle", "jurisdiction")),
            "income": ("Income and stability", ("kyc", "financial_capacity", "income")),
            "investable_assets": ("Investable assets", ("kyc", "financial_capacity", "assets")),
        },
    },
    "goals": {
        "label": "Objective",
        "hint": "Explicit financial objectives and funding plans. Most require a questionnaire; currently presented based on observable patterns and inferences.",
        "l2": {
            "goal_definition": ("Target definition", ("kyc", "goals_and_constraints", "goal_definition")),
            "funding_plan": ("Capital plan", ("kyc", "goals_and_constraints", "funding_plan")),
        },
    },
    "constraints": {
        "label": "Constraints",
        "hint": "Liquidity, protection, liabilities, and goal flexibility—these constraints determine tolerable risk and service approach.",
        "l2": {
            "flexibility": ("Target flexibility", ("kyc", "goals_and_constraints", "flexibility")),
            "liquidity": ("Liquidity requirements", ("kyc", "goals_and_constraints", "liquidity")),
            "protection": ("Coverage gap", ("kyc", "risk_and_protection", "protection")),
            "cashflow": ("Expenses and emergency funds", ("kyc", "financial_capacity", "cashflow")),
            "debt": ("Liabilities", ("kyc", "financial_capacity", "debt")),
        },
    },
    "risk_preference": {
        "label": "Risk preference",
        "hint": "Stated willingness, capacity to bear risk, risk required by the objective, and disclosed (behavioral) risk. The four must be stated separately.",
        "l2": {
            "willingness": ("Risk willingness (self-reported)", ("kyc", "risk_and_protection", "willingness")),
            "capacity": ("Risk tolerance", ("kyc", "risk_and_protection", "capacity")),
            "required_risk": ("Risk required for target", ("kyc", "risk_and_protection", "required_risk")),
            "scope": ("Applicable scope", ("kyc", "risk_and_protection", "scope")),
            "revealed_risk_behavior": ("Disclosed risk behavior", ("dim", "risk")),
        },
    },
    "emotion_and_psychology": {
        "label": "Sentiment and psychology",
        "hint": "Behavioral tendencies such as the disposition effect, chasing gains, and panic, as well as platform behavior observation signals. These are observation signals, not psychological diagnoses.",
        "l2": {
            "disposition_and_bias": ("Disposition effect and biases", ("dim", "emotion")),
            "loss_conditioned_attention": ("Attention to unrealized losses/unrealized gains", ("blk", "attention_capital", "loss_conditioned_attention")),
            "behavioral_signals": ("Class B behavior observation signal", ("bs",)),
        },
    },
    "investment_logic": {
        "label": "Investment logic",
        "hint": "Style, research depth, portfolio conviction, and the research→decision chain inferred from cross-stream evidence.",
        "l2": {
            "revealed_logic": ("Disclosed investment rationale", ("dim", "investment_logic")),
            "style": ("Style", ("kyc", "investment_logic", "style")),
            "fundamental_research": ("Research depth", ("kyc", "investment_logic", "fundamental_research")),
            "portfolio_belief": ("Portfolio conviction", ("kyc", "investment_logic", "portfolio_belief")),
        },
    },
    "app_behavior_analysis": {
        "label": "App behavior analysis",
        "hint": "Actual in-app usage: time and attention invested, engagement, research breadth, and the mismatch between attention and funds.",
        "l2": {
            "time_and_attention": ("Time and attention allocation", ("kyc", "investment_logic", "resource")),
            "engagement": ("Engagement", ("blk", "engagement", "")),
            "research_depth_and_breadth": ("Research depth and breadth", ("blk", "behavior_intent", "research")),
            "attention_vs_capital": ("Attention vs Capital Allocation", ("blk", "attention_capital", "alignment")),
        },
    },
    "personalization_and_rules": {
        "label": "Personalization and rules",
        "hint": "Financial advisor authorization scope, communication and presentation preferences, and service cadence - determining how the profile should be used.",
        "l2": {
            "authority": ("Decision authorization", ("kyc", "advisory_preferences", "authority")),
            "communication": ("Communication and presentation preferences", ("kyc", "advisory_preferences", "copy")),
            "service_rhythm": ("Service cadence", ("kyc", "advisory_preferences", "service_rhythm")),
        },
    },
    "investor_assessment": {
        "label": "Investor assessment",
        "hint": "Consistency of experience level, investment status, knowledge domain, self-reported/disclosed information, and objective assessment of the account and holdings.",
        "l2": {
            "experience": ("Experience and Knowledge Domain", ("dim", "investor_assessment")),
            "investment_status": ("Investment status", ("dim", "investor_assessment_status")),
            "preference_and_horizon": ("Preferences and horizon", ("dim", "preference")),
            "account": ("Account structure", ("kyc", "portfolio_state", "account")),
            "allocation": ("Asset allocation", ("kyc", "portfolio_state", "allocation")),
            "concentration": ("Concentration", ("kyc", "portfolio_state", "concentration")),
            "exposure": ("Exposure and leverage", ("kyc", "portfolio_state", "exposure")),
            "cashflow_and_experience": ("Cash flow and drawdown experience", ("kyc", "portfolio_state", "cashflow_and_experience")),
        },
    },
}

# The 10 L1 are FIXED. This is the presentation order, which is deliberately not
# the dict order: it reads as a biography (who they are -> what they want -> how
# they limit it -> how they actually behave -> how we serve them).
L1_ORDER = [
    "user_life_cycle",
    "background",
    "goals",
    "constraints",
    "risk_preference",
    "emotion_and_psychology",
    "investment_logic",
    "app_behavior_analysis",
    "personalization_and_rules",
    "investor_assessment",
]
# A drift here would silently change the contract with the frontend and the
# summary agent, so fail loudly at import time instead.
assert set(L1_ORDER) == set(L1_DEFS), f"L1_ORDER/L1_DEFS mismatch: {set(L1_ORDER) ^ set(L1_DEFS)}"

# ---------------------------------------------------------------------------
# NAACL27 fork: narrow the tree to the dimensions the financial data streams can support.
# The full definitions above are kept intact so the fork stays revertible; only
# the *effective* contract is filtered. See app/naacl_config.py for the rationale
# and the list of dropped dimensions.
# ---------------------------------------------------------------------------
try:
    from .naacl_config import KEEP_L1, KEEP_SUMMARY_GROUPS
except ImportError:                                    # standalone import
    from naacl_config import KEEP_L1, KEEP_SUMMARY_GROUPS

if KEEP_L1:
    _keep = set(KEEP_L1)
    _unknown = _keep - set(L1_DEFS)
    assert not _unknown, f"KEEP_L1 names unknown dimensions: {sorted(_unknown)}"
    L1_DEFS = {k: v for k, v in L1_DEFS.items() if k in _keep}
    L1_ORDER = [k for k in L1_ORDER if k in _keep]
    # Summary groups must only reference dimensions that still exist, or the
    # summary agent would be asked about a block it can never receive.
    SUMMARY_GROUPS = [[k for k in group if k in _keep]
                      for group in KEEP_SUMMARY_GROUPS]
    SUMMARY_GROUPS = [g for g in SUMMARY_GROUPS if g]

assert set(L1_ORDER) == set(L1_DEFS), f"L1_ORDER/L1_DEFS mismatch after filtering: {set(L1_ORDER) ^ set(L1_DEFS)}"
assert L1_ORDER, "no dimensions left after filtering"

# The production tree fixes this at 10. This fork fixes it at len(KEEP_L1) so a
# silent change to the kept set still fails loudly.
EXPECTED_L1_COUNT = len(KEEP_L1) if KEEP_L1 else 10
assert len(L1_ORDER) == EXPECTED_L1_COUNT, (
    f"L1 must stay fixed at {EXPECTED_L1_COUNT}, got {len(L1_ORDER)}")


# L3 statuses that mean "we do not know", not "the answer is zero"
_GAP_STATUS = {"gap", "insufficient"}

# `display` is the string the viewer prints as the fact, so it must ALWAYS be a
# human sentence. It must never be built with `str(value)`: several facts are
# mappings, so that prints a Python repr straight into the card. This was real —
# `research_depth` is a dict and `sessions_top` a list of dicts, and the tree
# rendered "{'start': '2026-05-08T01:24', 'primary_ticker': 'GH', ...}" to users.
# The division of labour is: `display` is prose, `value` keeps the raw structure
# (the viewer's collapsed 技术细节 section JSON-encodes `value`, which is the right
# place for it).
#
# The repr CONVERTER lives in app/evidence_glossary.py, not here. A second copy is
# how a third happens: this file used to carry one dead copy (with a different
# marker set from the live one) beside the live one, and the same conversion was
# being reinvented at the evidence boundary.


def _num(x, digits=1):
    return "—" if x is None else f"{round(x, digits):g}"


def _pct(x, digits=1):
    return "—" if x is None else f"{round(x * 100, digits):g}%"


def _human_display(value):
    """Human string for a value, or "" when there is no human form.

    Never falls back to `str(value)` for a mapping: returning "" lets the caller
    mark the leaf as structured (and keep the raw value in `value`) instead of
    dressing a repr up as prose.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if isinstance(value, list):
        return "、".join(p for p in (_human_display(v) for v in value) if p)
    if isinstance(value, dict):
        return ""
    return str(value)


def _research_depth_display(d):
    """`research_depth` is a mapping of counters — spell them out."""
    parts = []
    if d.get("sessions") is not None:
        parts.append(f"Research sessions {d['sessions']} times")
    if d.get("research_grade_sessions") is not None:
        parts.append(f"Among these, research-grade {d['research_grade_sessions']} sessions ({_pct(d.get('research_grade_session_share'))})")
    if d.get("median_session_events") is not None:
        parts.append(f"Median {d['median_session_events']} events / {_num(d.get('median_session_minutes'))} minutes per session")
    if d.get("max_session_events") is not None:
        parts.append(f"Maximum {d['max_session_events']} events in a single session")
    return "；".join(parts)


def _research_breadth_display(d):
    parts = []
    if d.get("unique_tickers") is not None:
        parts.append(f"Covered {d['unique_tickers']} instruments")
    if d.get("attention_hhi") is not None:
        parts.append(f"Attention HHI {_num(d['attention_hhi'], 3)} (lower means more dispersed)")
    if d.get("top_ticker_share") is not None:
        parts.append(f"Maximum single-instrument share {_pct(d['top_ticker_share'])}")
    if d.get("top_theme_share") is not None:
        parts.append(f"Maximum theme share {_pct(d['top_theme_share'])}")
    return "；".join(parts)


# NOTE the label: these are shares of BROWSE EVENTS by info_type, i.e. what the
# user looked at — not how deeply they analysed it. Presented as "浏览信息类型构成"
# so the number is not read as a research-quality claim.
_RESEARCH_STYLE_ZH = (("news_driven", "新闻"), ("fundamental_driven", "Fundamentals"),
                      ("technical_driven", "Technical analysis"), ("options_driven", "期权"),
                      ("social_driven", "Social"), ("price_check_share", "Quote lookup"))

_INTENT_STAGE_ZH = {"attention": "Attention", "research": "Research", "conviction": "Conviction",
                    "execution": "Trading", "holding": "Hold"}

_ACTION_ZH = {"start_new_position": "Open a position in a new instrument", "add_to_existing": "Add to an existing position",
              "trim_position": "Trim", "exit_position": "Exit", "hold": "Remain unchanged",
              "unknown": "Cannot determine"}


def _next_action_display(d):
    """The model's forward view as one sentence, with its confidence and basis."""
    if not isinstance(d, dict):
        return ""
    act = _ACTION_ZH.get(d.get("primary_action"), d.get("primary_action"))
    if not act:
        return ""
    parts = [f"The next step is more likely to be [{act}]"]
    if d.get("targets"):
        parts.append("May involve " + "、".join(str(t) for t in d["targets"][:5]))
    if d.get("horizon_days"):
        parts.append(f"Window approximately {d['horizon_days']} days")
    if d.get("confidence") is not None:
        try:
            parts.append(f"Confidence {float(d['confidence']):.0%}")
        except (TypeError, ValueError):
            pass
    if d.get("basis"):
        parts.append("Basis:" + "；".join(str(x) for x in d["basis"][:3]))
    if d.get("falsifier"):
        parts.append("Falsification conditions:" + str(d["falsifier"]))
    return "。".join(parts) + "。"


def _research_style_display(d):
    items = [(zh, d.get(k)) for k, zh in _RESEARCH_STYLE_ZH if d.get(k)]
    if not items:
        return ""
    return "Browsing information type composition:" + "、".join(f"{zh} {_pct(v)}" for zh, v in items)


def _sessions_display(rows, limit=3):
    """`sessions_top` is a list of mappings — render the top few as a sentence."""
    bits = []
    for r in rows[:limit]:
        if not isinstance(r, dict):
            continue
        seg = f"{r.get('primary_ticker') or '—'}（{str(r.get('start') or '')[:10]}"
        if r.get("events") is not None:
            seg += f", {r['events']} events"
        if r.get("minutes") is not None:
            seg += f" / {_num(r['minutes'])} minutes"
        stage = _INTENT_STAGE_ZH.get(r.get("max_intent_stage"))
        if stage:
            seg += f', reaching at most "{stage}"'
        bits.append(seg + "）")
    if not bits:
        return ""
    return f"Most concentrated {len(bits)} sessions:" + "；".join(bits)


# Fields whose value is a structure get an explicit human form. Anything absent
# here is NOT stringified — it is marked as structured and left in `value`.
_STRUCTURED_DISPLAY = {
    "research_depth": _research_depth_display,
    "research_breadth": _research_breadth_display,
    "research_style": _research_style_display,
    "sessions_top": _sessions_display,
    "next_action": _next_action_display,
}
STRUCTURED_DISPLAY_NOTE = "Structured metrics; see technical details"


def _display_for(key, value):
    """Prose for a leaf, or "" when the value has no human form."""
    fn = _STRUCTURED_DISPLAY.get(key)
    if fn is not None and isinstance(value, (dict, list)):
        try:
            return fn(value)
        except Exception:                                          # noqa: BLE001
            return ""
    return _human_display(value)


# The repr converter is imported from app/evidence_glossary (see `de_repr` there),
# aliased to the local name so the call sites below stay readable.
_de_repr = de_repr


def _clean_evidence(items):
    """Sanitise every evidence line. Evidence is quoted back to the reader."""
    return [_de_repr(x) if isinstance(x, str) else x for x in (items or [])]


def _leaf(key, label, value, display, status, evidence=None,
          confidence=None, data_needed=None, proxy_basis=None, caveat=None):
    """One L3 fact. Uniform shape regardless of which source it came from.

    `display` is coerced to a string (never a mapping, never None) so the viewer
    can rely on the contract, and so nothing downstream can accidentally receive a
    structure where it expects prose. `value` keeps the raw form for the collapsed
    technical section.

    `evidence` stays the raw citation (the audit trail); `evidence_plain` carries the
    same lines rendered for a person. Both are emitted because they answer different
    questions — "can I verify this?" and "what does this mean?" — and a citation
    like `account_snapshots[2026-09-06].holdings 按 instrument 分类聚合` answers only the
    first.
    """
    clean = _clean_evidence(evidence)
    _, notes = evidence_annotate(clean)
    return {
        "key": key, "label": label,
        "value": value,
        "display": display if isinstance(display, str) else "",
        "status": status,
        "evidence": clean,
        "evidence_plain": notes,
        "confidence": confidence,
        "data_needed": [d for d in (data_needed or []) if d],
        "proxy_basis": _de_repr(proxy_basis) if isinstance(proxy_basis, str) else proxy_basis,
        "caveat": _de_repr(caveat) if isinstance(caveat, str) else caveat,
    }


def _from_kyc_l2(kyc_tree, l1_key, l2_key):
    """Re-parent one KYC L2 block, preserving every leaf's status and gaps."""
    blk = (((kyc_tree or {}).get("l1") or {}).get(l1_key) or {}).get("l2") or {}
    l2 = blk.get(l2_key)
    if not l2:
        return None
    out = []
    for fk, f in (l2.get("features") or {}).items():
        status = FILLED if f.get("status") == "filled" else (
            PROXY if f.get("status") == "proxy" else GAP)
        out.append(_leaf(
            fk, f.get("label") or fk, f.get("value"), f.get("display"),
            status,
            evidence=f.get("evidence"),
            confidence=f.get("confidence"),
            data_needed=f.get("data_needed"),
            proxy_basis=f.get("proxy_basis"),
            caveat=f.get("caveat"),
        ))
    return {"source_l2": f"{l1_key}.{l2_key}", "leaves": out}


# Which dimension fields are FACTS (become L3 leaves) vs PROSE (becomes the L2
# summary fallback). Without this split, `narrative` and `key_signals` were turned
# into "leaves" and one dimension alone reported 21 facts, most of them prose.
DIM_LEAVES = {
    "risk": [("revealed_risk_behavior", "Disclosed risk behavior"),
             # v13: risk LEVEL says how much risk is carried; these say HOW, which
             # is what distinguishes two users who both read "aggressive".
             ("loss_response_style", "Response to losses"),
             ("drawdown_behavior", "Behavior during drawdowns"),
             ("position_sizing", "Position concentration"),
             ("leverage_usage", "Use of leveraged/derivative instruments"),
             ("risk_budget_stability", "Risk budget stability"),
             ("evidence_strength", "Evidence strength")],
    "investment_logic": [("primary_style", "Primary style"),
                         ("likely_styles", "Possible style"),
                         ("thematic_focus", "Attention themes"),
                         ("primary_signal_source", "Primary information basis"),
                         ("decision_speed", "Decision speed"),
                         ("conviction_style", "Position-opening method"),
                         ("thesis_durability", "Logic persistence"),
                         ("next_action", "Next-action prediction"),
                         ("evidence_strength", "Evidence strength")],
    "preferences": [("trading_style", "Trading method"),
                    ("horizon", "Holding period"),
                    ("stated_vs_revealed", "Stated-versus-behavior consistency"),
                    ("evidence_strength", "Evidence strength")],
    "emotion": [("disposition_effect", "Disposition effect"),
                ("chasing_momentum", "Tendency to chase rallies"),
                ("panic_patterns", "Panic mode"),
                ("evidence_strength", "Evidence strength")],
    "investor_assessment": [("experience_level", "Experience Level"),
                            ("investment_status", "Investment status"),
                            ("knowledge_domains", "Knowledge domain"),
                            ("lifecycle_stage", "Lifecycle stage"),
                            ("evidence_strength", "Evidence strength")],
}


def _from_dim(profile, dim_key, leaves_only=True):
    """Fact leaves + the dimension's narrative (used as the L2 summary fallback)."""
    d = (profile or {}).get(dim_key) or {}
    if not d:
        return None
    spec = DIM_LEAVES.get(dim_key)
    keys = spec if spec else [(k, k) for k in d]
    out = []
    for k, label in keys:
        v = d.get(k)
        if v in (None, "", [], {}) or k == "evidence_strength":
            continue
        text = _display_for(f"{dim_key}.{k}", v)
        # A structure with no human form is kept as a FACT but marked, rather than
        # disguised as prose by a repr.
        out.append(_leaf(
            f"{dim_key}.{k}", label, v,
            text or STRUCTURED_DISPLAY_NOTE,
            FILLED if text else PROXY,
            evidence=d.get("key_signals") or [],
            confidence=d.get("confidence"),
            caveat=None if text else "This field contains structured data; a human-readable description is not yet available",
        ))
    if d.get("unknown_fields"):
        out.append(_leaf(
            f"{dim_key}.unknown_fields", "Not yet confirmed", d["unknown_fields"],
            _human_display(d["unknown_fields"]), GAP,
            data_needed=d["unknown_fields"]))
    return {"source_dim": dim_key, "leaves": out,
            "narrative": d.get("narrative") or None}


def _from_build_attention(ac):
    """Loss-conditioned attention: the one place we compare attention by P&L state."""
    lc = (ac or {}).get("loss_conditioned_attention") or {}
    if not lc.get("available"):
        return None
    ev = [f"loss_attention_ratio={lc.get('loss_attention_ratio')}",
          f"share_on_losing={lc.get('share_on_losing')}",
          f"losing={lc.get('attention_on_losing_positions')}",
          f"winning={lc.get('attention_on_winning_positions')}"]
    if lc.get("sample_note"):
        return {"source_block": "attention_capital.loss_conditioned_attention",
                "leaves": [_leaf("loss_conditioned_attention", "Attention to unrealized losses/unrealized gains",
                                 None, lc["sample_note"], GAP,
                                 evidence=ev, data_needed=[lc["sample_note"]])]}
    return {"source_block": "attention_capital.loss_conditioned_attention",
            "leaves": [_leaf("loss_conditioned_attention", "Attention to unrealized losses/unrealized gains",
                             lc.get("interpretation"),
                             lc.get("interpretation") or "", FILLED, evidence=ev)]}


def _from_behavior_signals(kyc_tree):
    bs = ((kyc_tree or {}).get("behavior_signals") or {}).get("signals") or {}
    if not bs:
        return None
    out = []
    for k, v in bs.items():
        if v.get("status") == "filled":
            out.append(_leaf(k, v.get("label") or k, v.get("value"),
                             v.get("display") or "", FILLED,
                             evidence=v.get("evidence"),
                             confidence=v.get("confidence"),
                             # the signal's own limitation is an honesty marker,
                             # so it travels with the fact
                             caveat=v.get("limitation")))
        else:
            out.append(_leaf(k, v.get("label") or k, None,
                             v.get("display") or v.get("description") or "", GAP,
                             data_needed=v.get("data_needed")))
    return {"source_block": "kyc_tree.behavior_signals", "leaves": out}


def _from_alignment(ac):
    al = (ac or {}).get("attention_capital_alignment") or {}
    if not al.get("available"):
        return None
    sc = al.get("state_counts") or {}
    out = [_leaf("alignment_states", "Four types of relationships between attention and holdings",
                 sc, "；".join(f"{k}: {v}" for k, v in sc.items()), FILLED,
                 evidence=[f"dominant_state={al.get('dominant_state')}",
                           f"attention_gt_capital={al.get('attention_gt_capital')}",
                           f"capital_gt_attention={al.get('capital_gt_attention')}"])]
    if al.get("dominant_state"):
        out.append(_leaf("dominant_state", "Primary relationship", al["dominant_state"],
                         al["dominant_state"], FILLED))
    return {"source_block": "attention_capital.alignment", "leaves": out}


def _from_engagement(snap):
    ev = (snap or {}).get("behaviors_6m") or []
    acct = (snap or {}).get("account_daily") or []
    if not ev and not acct:
        return None
    leaves = []
    if ev:
        days = {str(e.get("event_date") or "")[:10] for e in ev}
        leaves.append(_leaf("event_count", "Number of behavioral events", len(ev), f"{len(ev)} items", FILLED))
        leaves.append(_leaf("active_days", "Active days", len(days), f"{len(days)} days", FILLED,
                            evidence=[f"events={len(ev)}"]))
    if acct:
        leaves.append(_leaf("account_coverage_days", "Account data coverage", len(acct),
                            f"{len(acct)} days", FILLED))
    return {"source_block": "snapshot", "leaves": leaves}


def _from_research(bi):
    if not (bi or {}).get("available"):
        return None
    out = []
    for key, label, path in (("depth", "Research depth", "research_depth"),
                             ("breadth", "Research breadth", "research_breadth"),
                             ("style", "Research composition", "research_style"),
                             ("sessions", "Research session", "sessions_top")):
        v = (bi or {}).get(path)
        if not v:
            continue
        text = _display_for(path, v)
        out.append(_leaf(key, label, v,
                         text or STRUCTURED_DISPLAY_NOTE,
                         FILLED if text else PROXY,
                         evidence=["behavior_intent." + path],
                         caveat=None if text else "This field contains structured data; a human-readable description is not yet available"))
    return {"source_block": "behavior_intent.research", "leaves": out}


def _from_alignment_block(ac):
    return _from_alignment(ac)


# ---- L2 resolution ---------------------------------------------------------

def _resolve(spec, ctx):
    kind = spec[0]
    if kind == "kyc":
        return _from_kyc_l2(ctx["kyc_tree"], spec[1], spec[2])
    if kind == "dim":
        dim = spec[1]
        if dim == "investor_assessment_status":
            d = (ctx["profile"].get("investor_assessment") or {})
            if not d:
                return None
            v = {k: d[k] for k in ("investment_status", "notes") if d.get(k)}
            if not v:
                return None
            return {"source_dim": "investor_assessment",
                    "leaves": [_leaf("investment_status", "Investment status",
                                     d.get("investment_status"),
                                     _human_display(d.get("investment_status")),
                                     FILLED, evidence=d.get("key_signals") or [])]}
        if dim == "investor_assessment_lifecycle":
            d = (ctx["profile"].get("investor_assessment") or {})
            v = d.get("lifecycle_stage")
            if not v or v == "unknown":
                return {"source_dim": "investor_assessment",
                        "leaves": [_leaf("lifecycle_stage.assessed", "Assessed life cycle", None,
                                         "Current data is insufficient to determine life stage", GAP,
                                         data_needed=["KYC questionnaire: life stage"])]}
            return {"source_dim": "investor_assessment",
                    "leaves": [_leaf("lifecycle_stage.assessed", "Assessed life cycle", v,
                                     _human_display(v), FILLED,
                                     evidence=(d.get("key_signals") or []))]}
        if dim == "preference":
            return _from_dim(ctx["profile"], "preferences")
        return _from_dim(ctx["profile"], dim)
    if kind == "bs":
        return _from_behavior_signals(ctx["kyc_tree"])
    if kind == "blk":
        blk, path = spec[1], spec[2]
        if blk == "attention_capital" and path == "loss_conditioned_attention":
            return _from_build_attention(ctx["attention_capital"])
        if blk == "attention_capital" and path == "alignment":
            return _from_alignment_block(ctx["attention_capital"])
        if blk == "engagement":
            return _from_engagement(ctx["snap"])
        if blk == "behavior_intent" and path == "research":
            return _from_research(ctx.get("behavior_intent"))
    return None


def build_profile_tree(profile, kyc_tree=None, attention_capital=None,
                       behavior_intent=None, snap=None, supported_only=True,
                       evidence=None):
    """Assemble the tree. Deterministic: no LLM, safe to run on every refresh.

    `supported_only` (the default, and what production stores) keeps ONLY facts
    that have evidence behind them: `filled` (straight from the data) and `proxy`
    (an inference whose basis is stated). A `gap` is not a conclusion — it is the
    ABSENCE of one — so it is neither shown nor persisted. Missing facts are
    recorded as COUNTS, never as content, so the omission stays disclosed without
    dressing an unknown up as a finding.

    Dropping gaps costs nothing: they are a deterministic function of what is
    absent, so they can be regenerated on demand. `supported_only=False` exists for
    exactly that — the structural tests use it to prove that re-parenting the KYC
    labels loses nothing.

    `evidence` is the full deterministic package. When supplied, `app/l2_taxonomy`
    contributes the curated sub-dimensions (goals -> time horizon / return
    objective / …, risk -> observed capacity / structural exposure / …) computed
    from behavior_events + trade_executions + account_snapshots. Without it the tree keeps only its KYC/dimension
    L2, which is the pre-v2.8 shape: 6 of the 10 L1 had no subdivision at all.
    """
    ctx = {"profile": profile or {}, "kyc_tree": kyc_tree or {},
           "attention_capital": attention_capital or {},
           "behavior_intent": behavior_intent or {}, "snap": snap or {},
           "evidence": evidence or {}}
    l1_out = {}
    counts = {"filled": 0, "proxy": 0, "gap": 0}
    # Curated sub-dimensions, built once and merged per L1 below. A failure inside
    # must degrade DEPTH only, never the whole tree.
    try:
        from .l2_taxonomy import build_l2
        extra_l2 = build_l2(evidence or {}, dims=(profile or {})) if evidence else {}
    except Exception as e:                                            # noqa: BLE001
        extra_l2 = {}
        counts["l2_taxonomy_error"] = f"{type(e).__name__}: {e}"
    for l1_key in L1_ORDER:
        spec = L1_DEFS.get(l1_key)
        if not spec:
            continue
        l2_out = {}
        l1_all = []
        for l2_key, (l2_label, source) in spec["l2"].items():
            got = _resolve(source, ctx)
            if not got:
                # The source itself is absent. Recorded as a gap so that, under
                # supported_only, it never reaches the output.
                leaves = [_leaf(f"{l2_key}.unavailable", l2_label, None,
                                "The source data is missing", GAP,
                                data_needed=["Required data stream is not connected or is empty"])]
                src = "missing"
            else:
                leaves = got["leaves"]
                src = got.get("source_l2") or got.get("source_dim") or got.get("source_block")
            for lf in leaves:
                counts[lf["status"]] = counts.get(lf["status"], 0) + 1
            # Boundary guard: person-identifying content never enters the profile,
            # whatever source produced it. Removing the one channel that carried
            # it is not enough — the same defect can arrive from a new upstream or
            # from a model writing prose, and the fact that it shipped once already
            # says filtering at the boundary is worth the check.
            clean = []
            for lf in leaves:
                reason = (find_pii(lf.get("label")) or find_pii(lf.get("display"))
                          or find_pii(lf.get("value")) or find_pii(lf.get("evidence"))
                          or find_pii(lf.get("caveat")))
                if reason:
                    counts["pii"] = counts.get("pii", 0) + 1
                    continue
                clean.append(lf)
            leaves = clean
            l1_all.extend(leaves)
            shown = leaves if not supported_only else [
                x for x in leaves if x["status"] != GAP]
            if not shown:
                # Nothing evidenced here: the L2 is omitted entirely rather than
                # rendered as an empty bucket.
                continue
            filled = sum(1 for x in shown if x["status"] == FILLED)
            prox = sum(1 for x in shown if x["status"] == PROXY)
            l2_out[l2_key] = {
                "label": l2_label, "source": src,
                "leaves": shown,
                # deterministic prose fallback: the source dimension's own
                # narrative. The LLM summary overwrites it when available, so
                # the UI is never left with an empty explanation.
                "summary": (got or {}).get("narrative"),
                "summary_kind": "source_narrative" if (got or {}).get("narrative") else None,
                "filled": filled, "proxy": prox, "gap": 0, "total": len(shown),
                "status": ("filled" if prox == 0 and filled else
                           "partial" if (filled or prox) else "gap"),
            }
        # --- curated sub-dimensions (v2.8) ------------------------------------
        # Merged HERE, after the KYC/dimension L2, so they EXTEND that structure
        # rather than replacing it. They go through exactly the same discipline:
        # the PII boundary guard, gap-dropping, and the counts. A different path
        # with looser rules would be the way a leak or a zero-fill gets in.
        for l2_key, blk in (extra_l2.get(l1_key) or {}).items():
            if l2_key in l2_out:
                # A key collision means one of the two blocks is DROPPED. Curated
                # keys are meant to be distinct from the KYC ones; when they are not,
                # the curated facts vanish with no trace — the exact silent-loss
                # pattern this codebase keeps having to fix. Recorded, not ignored.
                counts.setdefault("l2_key_collisions", [])
                counts["l2_key_collisions"].append(f"{l1_key}.{l2_key}")
                continue
            leaves = blk.get("leaves") or []
            for lf in leaves:
                counts[lf["status"]] = counts.get(lf["status"], 0) + 1
            clean = []
            for lf in leaves:
                reason = (find_pii(lf.get("label")) or find_pii(lf.get("display"))
                          or find_pii(lf.get("value")) or find_pii(lf.get("evidence"))
                          or find_pii(lf.get("caveat")))
                if reason:
                    counts["pii"] = counts.get("pii", 0) + 1
                    continue
                clean.append(lf)
            l1_all.extend(clean)
            shown = clean if not supported_only else [
                x for x in clean if x["status"] != GAP]
            if not shown:
                continue
            filled = sum(1 for x in shown if x["status"] == FILLED)
            prox = sum(1 for x in shown if x["status"] == PROXY)
            l2_out[l2_key] = {
                "label": blk.get("label") or l2_key,
                "source": "l2_taxonomy",
                "hint": blk.get("hint"),
                "leaves": shown,
                "summary": blk.get("hint"),
                "summary_kind": "derived_hint" if blk.get("hint") else None,
                "filled": filled, "proxy": prox, "gap": 0, "total": len(shown),
                "status": ("filled" if prox == 0 and filled else
                           "partial" if (filled or prox) else "gap"),
            }
        l1_shown = [x for l2 in l2_out.values() for x in l2["leaves"]]
        filled = sum(1 for x in l1_shown if x["status"] == FILLED)
        prox = sum(1 for x in l1_shown if x["status"] == PROXY)
        l1_out[l1_key] = {
            "label": spec["label"], "hint": spec["hint"],
            "l2": l2_out, "summary": None,
            "filled": filled, "proxy": prox, "gap": 0, "total": len(l1_shown),
            "omitted": len(l1_all) - len(l1_shown),
            # An L1 with no supported fact still appears. The 10-L1 skeleton is a
            # FIXED contract and is what makes two users comparable, so it is kept
            # as an empty bucket rather than dropped — otherwise "this dimension
            # was never assessed" and "it was assessed and nothing was found"
            # become indistinguishable.
            "status": ("filled" if prox == 0 and filled else
                       "partial" if (filled or prox) else "gap"),
        }
    total = sum(counts.values())
    shown_total = counts["filled"] + counts["proxy"] if supported_only else total
    omitted = total - shown_total
    pii_dropped = counts.get("pii", 0)
    tree = {
        "version": tree_version_stamp(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "l1_order": list(L1_ORDER),
        "l1": l1_out,
        "supported_only": bool(supported_only),
        "summary": {
            "filled": counts["filled"], "proxy": counts["proxy"], "gap": counts["gap"],
            "pii_dropped": pii_dropped,
            "total": total, "shown": shown_total, "omitted": omitted,
            "covered": shown_total,
            "fill_rate": round(shown_total / total, 3) if total else 0.0,
            "direct_fill_rate": (round(counts["filled"] / shown_total, 3)
                                 if shown_total else 0.0),
            "note": (f"Only conclusions supported by evidence are shown: {counts['filled']} derived directly from data, and {counts['proxy']} are evidence-based indirect inferences (basis of inference labeled)."
                     + (f"{omitted} additional items lack supporting data and are excluded from conclusions and storage."
                        if omitted else "")
                     # Wording matters here: the note must not contain the very words
                     # the PII guard scans for, or the profile flags its own disclosure.
                     + (f"Removed {pii_dropped} items involving personally identifiable information."
                        if pii_dropped else "")
                     + "L1 is the fixed skeleton; L2/L3 appear only when supporting facts exist."),
        },
        "summaries_generated": False,
    }
    from .english_output import englishize
    return englishize(tree)


def tree_for_prompt(tree):
    """Compact payload for the summary LLM calls: L3 facts only, no prose yet."""
    out = {}
    for l1_key, l1 in (tree.get("l1") or {}).items():
        out[l1_key] = {
            "label": l1["label"],
            "hint": l1["hint"],
            "l2": {
                l2_key: {
                    "label": l2["label"],
                    "facts": [
                        {"label": f["label"],
                         "value": f["display"] or f["value"],
                         "status": f["status"]}
                        for f in l2["leaves"]
                    ],
                }
                for l2_key, l2 in l1["l2"].items()
            },
        }
    return out


def apply_summaries(tree, payload):
    """Attach model-written L1/L2 summaries. Ignores anything it cannot place.

    Every summary is passed through the PII scrubber. The guard on leaves removes
    the FACTS, but prose is generated and can reintroduce what the data no longer
    contains — a narrative that says "用户自述为 55 岁美国人" is a real example of
    exactly that. Scrubbing whole sentences (rather than dropping the summary) keeps
    the analytical content and loses only the sentence that carried person data.
    """
    if not isinstance(payload, dict):
        return tree
    applied = 0
    for l1_key, got in payload.items():
        l1 = (tree.get("l1") or {}).get(l1_key)
        if not l1 or not isinstance(got, dict):
            continue
        if got.get("summary"):
            text = pii_scrub(str(got["summary"]).strip())
            if text:
                l1["summary"] = text
                l1["summary_kind"] = "llm"
                applied += 1
        for l2_key, s in (got.get("l2") or {}).items():
            l2 = l1["l2"].get(l2_key)
            if l2 and s:
                text = pii_scrub(str(s).strip())
                if text:
                    l2["summary"] = text
                    l2["summary_kind"] = "llm"
    tree["summaries_generated"] = applied > 0
    return tree


def tree_digest(tree):
    """Small view for prompts/UI headers."""
    if not tree:
        return {"available": False}
    return {
        "available": True,
        "version": tree.get("version"),
        "summary": tree.get("summary"),
        "l1": {k: {"label": v["label"], "status": v["status"],
                   "summary": v.get("summary")}
               for k, v in (tree.get("l1") or {}).items()},
        "summaries_generated": tree.get("summaries_generated"),
    }
