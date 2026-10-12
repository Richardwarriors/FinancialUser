"""Stable JSON contracts shared by profile export and rolling prediction."""
from __future__ import annotations

PROFILE_SCHEMA_VERSION = "naacl27_point_in_time_profile_en_v2"
# The published profile drops the evidence/presentation audit layers: the paper
# reports conclusions, not stored evidence. `PUBLIC_PROFILE_SCHEMA_VERSION`
# marks that narrower contract so a full internal profile can never be
# mistaken for a published one.
PUBLIC_PROFILE_SCHEMA_VERSION = "naacl27_public_profile_en_v3"
PROFILE_DIMENSIONS = ("risk_preference", "emotion_and_psychology",
                      "investment_logic", "app_behavior_analysis",
                      "investor_assessment")
SAMPLE_SCHEMA_VERSION = "naacl27_rolling_prediction_en_v7"

# Trait fields the published profile must not carry.  The harness runs a rule
# engine that forecasts the next capital action, and it writes the answer into
# `investment_logic.next_action` in the SAME vocabulary as the position-action
# label ("trim_position", "add_to_existing", ...).  It is computed only from
# pre-cutoff evidence, so it is not temporal leakage -- but publishing it inside
# the evidence would let a predictor copy one field and score the rule engine
# instead of reasoning over behaviour, which is exactly the claim the dataset
# exists to test.  It is archived as a sample-level baseline instead.
BANNED_PUBLISHED_TRAIT_FIELDS = ("next_action",)
BANNED_TRAIT_KEY_SUBSTRINGS = ("forecast", "prediction")

REQUIRED_PROFILE_KEYS = (
    "user_id", "profile_version", "cutoff_et", "profile_date",
    "base_profile", "structured", "data_quality", "provenance",
)


def published_forecast_fields(node, path="profile") -> list[str]:
    """Return paths of forbidden rule-forecast fields at any nesting depth."""
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            lowered = str(key).lower()
            if key in BANNED_PUBLISHED_TRAIT_FIELDS or any(
                    marker in lowered for marker in BANNED_TRAIT_KEY_SUBSTRINGS):
                out.append(child)
            out.extend(published_forecast_fields(value, child))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(published_forecast_fields(value, f"{path}[{i}]"))
    return out


def public_profile_contamination(node, path="profile") -> list[str]:
    """Return structural and semantic action-forecast leaks with JSON paths."""
    from .forecast_scrub import find_leaks

    return published_forecast_fields(node, path) + find_leaks(node, path)


def validate_profile(doc: dict) -> dict:
    if not isinstance(doc, dict):
        raise ValueError("profile document must be an object")
    if doc.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported profile schema {doc.get('schema_version')!r}; "
            f"expected {PROFILE_SCHEMA_VERSION!r}")
    missing = [k for k in REQUIRED_PROFILE_KEYS if k not in doc or doc[k] is None]
    if missing:
        raise ValueError(f"profile document missing required keys: {missing}")
    base = doc["base_profile"]
    if not isinstance(base, dict) or not base.get("summary"):
        raise ValueError("base_profile.summary is required")
    if not isinstance(doc["structured"], dict) or not doc["structured"]:
        raise ValueError("structured profile dimensions are required")
    return doc


def validate_public_profile(doc: dict) -> dict:
    """The published profile: one summary plus a summary for each of five dimensions."""
    if not isinstance(doc, dict):
        raise ValueError("public profile must be an object")
    if doc.get("schema_version") != PUBLIC_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported public profile schema "
                         f"{doc.get('schema_version')!r}; "
                         f"expected {PUBLIC_PROFILE_SCHEMA_VERSION!r}")
    for banned in ("evidence_summary", "presentation"):
        if banned in doc:
            raise ValueError(f"{banned} must not be published")
    for required in ("user_id", "profile_version", "cutoff_et", "profile_date",
                     "base_profile", "structured"):
        if not doc.get(required) and doc.get(required) != 0:
            raise ValueError(f"public profile missing {required}")
    base = doc["base_profile"]
    if not base.get("summary"):
        raise ValueError("base_profile.summary is required")
    dims = base.get("dimensions") or {}
    missing = [d for d in PROFILE_DIMENSIONS if not (dims.get(d) or "").strip()]
    if missing:
        raise ValueError(f"base_profile.dimensions is missing summaries for {missing}")
    contaminated = public_profile_contamination(doc)
    if contaminated:
        raise ValueError(
            "public profile contains rule-engine forecast contamination at "
            f"{contaminated[:10]}; it belongs only in the sample baselines block")
    return doc


def validate_sample(doc: dict) -> dict:
    if not isinstance(doc, dict):
        raise ValueError("rolling sample must be an object")
    if doc.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError("rolling sample schema mismatch")
    required = ("sample_id", "user_id", "cutoff_et", "prediction_date",
                "profile_snapshot", "recent_7d", "market_7d", "targets",
                "observation_coverage", "leakage_audit", "baselines", "provenance")
    missing = [k for k in required if k not in doc]
    if missing:
        raise ValueError(f"rolling sample missing required keys: {missing}")
    if not (doc.get("leakage_audit") or {}).get("clean"):
        raise ValueError("rolling sample failed leakage audit")
    trade = (doc.get("targets") or {}).get("trade_decision")
    if not isinstance(trade, dict) or "trade_activity" not in trade:
        raise ValueError("rolling sample missing hierarchical trade_decision target")
    activity = trade.get("trade_activity")
    sides = trade.get("trade_sides")
    if activity not in ("trade", "no_trade", "unknown"):
        raise ValueError(f"invalid trade_activity {activity!r}")
    if sides not in ("buy_only", "sell_only", "buy_and_sell",
                     "not_applicable", "unknown"):
        raise ValueError(f"invalid trade_sides {sides!r}")
    if activity == "no_trade" and (sides != "not_applicable" or
                                    trade.get("buy_tickers") or
                                    trade.get("sell_tickers")):
        raise ValueError("no_trade target must have no sides or ticker targets")
    return doc
