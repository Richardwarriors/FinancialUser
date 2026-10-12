"""LLM extraction for the judgement-based KYC features.

Only the 20 features marked `llm` in app/kyc_catalog.py reach this module. The
31 `unavailable` features never do: they require data we do not have, and asking
a model to infer annual income from browsing behaviour would be fabrication.

Grouped into 3 calls rather than 20 so the shared evidence pack is sent once per
theme and per-call schema stays focused:

    goals_risk   goals_and_constraints + risk_and_protection          (4 features)
    logic        investment_logic                                     (7 features)
    context      advisory_preferences + financial_capacity + lifecycle (9 features)

Hard rules enforced by prompt and by post-validation:
  * every feature must cite `evidence` strings that literally exist in the
    evidence blob, same grounding rule as the profilers;
  * a feature the evidence cannot support must be returned as
    value="insufficient_data" with `missing` explaining what is needed —
    never a plausible-sounding guess;
  * no advice, no suitability conclusions, no forecasts.
"""
import json
import os
import re

from pydantic import BaseModel, Field, create_model

from .kyc_catalog import CATALOG

MAX_TOKENS = int(os.environ.get("LUNA_MAX_TOKENS", "4000"))
KYC_PROMPT_VERSION = "kyc-v1"

# value types per feature, so the generated schema can validate shapes
_NUM = "number"
_CAT = "category"
_TXT = "text"

FEATURE_TYPE = {
    # goals_and_constraints
    "investment_horizon_framework": _CAT,
    "minimum_liquidity_reserve": _CAT,
    # risk_and_protection
    "maximum_uncomfortable_drawdown": _NUM,
    "risk_capacity": _CAT,
    # investment_logic
    "preferred_investment_style": _CAT,
    "fundamental_analysis_depth": _CAT,
    "investment_domain_expertise": _CAT,
    "thesis_evidence_requirement": _CAT,
    "valuation_sensitivity": _CAT,
    "diversification_belief": _CAT,
    "time_and_attention_budget": _CAT,
    # advisory_preferences
    "advisor_autonomy_preference": _CAT,
    "advisory_language_preference": _CAT,
    "explanation_depth_preference": _CAT,
    "communication_tone_preference": _CAT,
    "output_format_preference": _CAT,
    "portfolio_review_cadence": _CAT,
    # financial_capacity
    "emergency_fund_runway": _CAT,
    # jurisdiction_and_lifecycle
    "occupation_industry": _CAT,
    "wealth_lifecycle_stage": _CAT,
}

# Which evidence streams each group may see. Mirrors the profile permission
# matrix: attention/self-report must not be laundered into capacity claims.
GROUP_STREAMS = {
    "goals_risk": ["position", "asset", "trade", "behavior_intent", "stated", "conversation"],
    "logic": ["behavior", "behavior_intent", "trade", "behavior_trade_match",
              "position", "asset", "conversation", "stated"],
    "context": ["behavior_intent", "position", "asset", "trade", "conversation", "stated"],
}
GROUP_L1 = {
    "goals_risk": ["goals_and_constraints", "risk_and_protection"],
    "logic": ["investment_logic"],
    "context": ["advisory_preferences", "financial_capacity", "jurisdiction_and_lifecycle"],
}

# Rolling profile configuration: `CATALOG` is filtered to the blocks the retained dimensions
# resolve against, so a group naming a dropped block would raise KeyError and
# take the whole KYC layer down with it (observed: `KeyError:
# 'goals_and_constraints'` logged as `[harness_kyc]`, tree then missing).
# Filtering here keeps the grouping definitions intact for a revert while making
# the live groups a function of what actually exists.
GROUP_L1 = {g: [l1 for l1 in l1s if l1 in CATALOG] for g, l1s in GROUP_L1.items()}
GROUP_L1 = {g: l1s for g, l1s in GROUP_L1.items() if l1s}
assert GROUP_L1, "no KYC group has any live block left"

# Keep the stream policy in step with the surviving groups: a group that no
# longer exists must not keep a stream permission key around, or a future
# re-add of the block would silently inherit a policy nobody re-reviewed.
GROUP_STREAMS = {g: s for g, s in GROUP_STREAMS.items() if g in GROUP_L1}

SYSTEM = (
    "Complete KYC features for an investment-advisory system using only the supplied "
    "evidence package. Never invent information. For each feature: value is the supported "
    "judgment, or insufficient_data when evidence is inadequate; missing states exactly "
    "what additional data is required; evidence contains one to three literal references "
    "to supplied fields or values; confidence is 0-1; narrative is one or two concise "
    "English sentences explaining the basis and uncertainty. Keep stated claims separate "
    "from revealed behavior. Browsing is not evidence of risk capacity, income, or protection. "
    "Do not provide advice, suitability conclusions, or predictions. ALL natural-language "
    "output must be English. Return schema-compliant JSON only."
)

_GROUP_HINT = {
    "goals_risk": "This group focuses on goals, time horizon, liquidity, and risk. Risk willingness and risk capacity require user confirmation; without self-description or assessment data, confirmed conclusions are always insufficient_data; indirect judgments can be made based on drawdowns actually tolerated, investable asset size, and holding period, but they must be identified as indirect evidence.",
    "logic": "This group focuses on investment logic and research methods. behavior_intent is the core evidence: research_style distinguishes news/technical/fundamental/options-driven activity; research_depth / research_breadth / exploration_vs_exploitation reflect research investment and breadth; distinguish \"viewing a lot\" from \"researching deeply\": timeframe_switches, view_company, view_diagnosis, and view_earnings represent fundamental research, while view_quote only indicates viewing quotes.",
    "context": "This group focuses on preferences for communication with investment advisors, financial capability, and life-cycle background. Communication features can only come from the language, phrasing, and formatting requirements in conversations conversation titles; insufficient_data is required when there are too few title samples; most financial capability features require questionnaire data, and without data they are always insufficient_data; life cycle and industry can only come from the user's self-description or explicit mentions in conversations and must not be inferred from asset size.",
}


def _spec(l1, feature):
    node = CATALOG[l1]["l2"]
    for l2, l2node in node.items():
        if feature in l2node["features"]:
            return l2, l2node["features"][feature]
    return None, None


def group_features(group):
    out = []
    for l1 in GROUP_L1[group]:
        for l2, l2node in CATALOG[l1]["l2"].items():
            for key, spec in l2node["features"].items():
                label, cov, streams, needed = spec
                if cov == "llm":
                    out.append({"l1": l1, "l2": l2, "key": key, "label": label,
                                "type": FEATURE_TYPE.get(key, _CAT)})
    return out


class FeatureOut(BaseModel):
    value: str = Field(description="Supported English judgment, or insufficient_data")
    evidence: list[str] = Field(default_factory=list, description="Literal supplied evidence references in English")
    confidence: float = Field(ge=0, le=1)
    narrative: str = Field(description="One or two concise English sentences")
    missing: str = Field(default="", description="Missing data required when value is insufficient_data")


def _narr_model(name, keys):
    """Explicit per-key narrative model.

    A bare `dict` field is rejected by the structured-output schema validator
    (it requires declared properties), so the narrative maps are generated as
    real models with one field per L1/L2 key.

    Keys that collide with pydantic BaseModel attributes (`copy`, `dict`, ...)
    are suffixed, otherwise the generated field shadows the base attribute and
    pydantic warns (and the attribute becomes the field).
    """
    reserved = set(dir(BaseModel))
    mapping = {}
    fields = {}
    for k in keys:
        safe = f"{k}_narr" if k in reserved else k
        mapping[safe] = k
        fields[safe] = (str, Field(default="", description=f"English description of {k}"))
    return create_model(name, **fields), mapping


def _schema(group, feats, l1_keys):
    l2_keys = [l2 for l1 in l1_keys for l2 in CATALOG[l1]["l2"]]
    l2_model, l2_map = _narr_model(f"L2_{group}", l2_keys)
    l1_model, l1_map = _narr_model(f"L1_{group}", l1_keys)
    fields = {f["key"]: (FeatureOut, Field(description=f["label"])) for f in feats}
    fields["l2_narratives"] = (l2_model, Field(description="One English description per L2"))
    fields["l1_narratives"] = (l1_model, Field(description="One English overview per L1"))
    return create_model(f"Kyc_{group}", **fields), (l2_map, l1_map)


def _permitted(evidence, group):
    return {k: v for k in GROUP_STREAMS[group]
            for k, v in [(k, evidence.get(k))] if isinstance(v, dict) and v.get("available")}


def _numeric(value):
    m = re.search(r"-?\d+(\.\d+)?", str(value or ""))
    return float(m.group(0)) if m else None


def fill_group(group, evidence, deterministic=None, timeout_note=""):
    """Fill one group's llm features. Returns (features_dict, l2_narr, l1_narr, source)."""
    from .profilers import _client

    feats = group_features(group)
    l1_keys = GROUP_L1[group]
    permitted = _permitted(evidence, group)
    if deterministic:
        # the already-computed deterministic numbers are legitimate evidence
        permitted = {**permitted, "computed_portfolio_metrics": deterministic}

    client = _client()
    if client is None:
        return {}, {}, {}, "rule"

    l2_keys = [l2 for l1 in l1_keys for l2 in CATALOG[l1]["l2"]]
    payload = {
        "features_to_fill": [{k: f[k] for k in ("key", "label", "type")} for f in feats],
        "l2_to_describe": l2_keys,
        "l1_to_summarise": l1_keys,
        "evidence": permitted,
    }
    try:
        schema, maps = _schema(group, feats, l1_keys)
        from .profilers import resolve_model
        from .token_meter import record
        model = resolve_model("assessment")     # KYC inference is structured, not deep
        raw_run = client.with_structured_output(schema, include_raw=True)
        res = raw_run.invoke([
            ("system", SYSTEM + "\n" + _GROUP_HINT.get(group, "")),
            ("user", json.dumps(payload, ensure_ascii=False) + timeout_note),
        ])
        record("kyc", model, getattr(res.get("raw"), "usage_metadata", None))
        out = res.get("parsed")
        if out is None:
            return {}, {}, {}, "rule:llm_parse_failed"
        d = out.model_dump()
    except Exception as e:                                            # noqa: BLE001
        return {}, {}, {}, f"rule:llm_error:{type(e).__name__}"

    l2_map, l1_map = maps
    l2_raw = d.get("l2_narratives") or {}
    l1_raw = d.get("l1_narratives") or {}
    l2_narr = {orig: (l2_raw.get(safe) or "") for safe, orig in l2_map.items()}
    l1_narr = {orig: (l1_raw.get(safe) or "") for safe, orig in l1_map.items()}

    blob = json.dumps(permitted, ensure_ascii=False)
    result = {}
    for f in feats:
        r = d.get(f["key"]) or {}
        val = r.get("value", "insufficient_data")
        if f["type"] == _NUM and val not in (None, "insufficient_data"):
            num = _numeric(val)
            val = num if num is not None else "insufficient_data"
        sig = [s for s in (r.get("evidence") or [])
               if re.search(r"[a-zA-Z_\u4e00-\u9fff]", str(s))
               and any(tok in blob for tok in re.findall(r"[A-Za-z_]{3,}", str(s)))]
        result[f["key"]] = {
            "value": val,
            "evidence": sig,
            "confidence": r.get("confidence", 0.0),
            "narrative": re.sub(r"\s+", " ", str(r.get("narrative") or "")).strip(),
            "missing": str(r.get("missing") or "").strip(),
            "grounded": bool(sig) or val == "insufficient_data",
        }
    return result, l2_narr, l1_narr, "llm"


GROUP_ORDER = ["goals_risk", "logic", "context"]

# Rolling profile configuration: run only the groups that still have a live KYC block. Without
# this, `fill_group("context")` raises `KeyError: 'context'` and the whole KYC
# call aborts — the layer silently degrades to "no KYC features" while the run
# still reports success.
GROUP_ORDER = [g for g in GROUP_ORDER if g in GROUP_L1]
assert GROUP_ORDER, "no KYC group left to run"


def build_kyc_llm_features(evidence, deterministic=None):
    """Run all groups. Returns (features, l2_narr, l1_narr, sources)."""
    feats, l2n, l1n, sources = {}, {}, {}, {}
    for g in GROUP_ORDER:
        f, l2, l1, src = fill_group(g, evidence, deterministic)
        feats.update(f)
        l2n.update(l2)
        l1n.update(l1)
        sources[g] = src
    return feats, l2n, l1n, sources
