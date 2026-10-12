"""Profiler Agents: LLM-driven dimension analysis with strict schemas.

Each agent sees ONLY the evidence its policy allows (information boundary),
returns structured JSON, and falls back to a deterministic rule profile when
the LLM is unavailable or produces schema-invalid output.
"""
import json
import os
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

MODEL_VERSION = os.environ.get("LUNA_MODEL", "gpt-5.6-luna")
# v12: removed the `stated` evidence channel (person-identifying demographic text)
# from POLICY for all five dimensions, and with it the two rule heuristics that read
# it. The dimension OUTPUT therefore changes for any user whose self-report was
# present, which is why the producer version moves with it — the evidence fingerprint
# already catches the input change, but the rule-layer behaviour change would not be
# captured by anything else.
PROMPT_VERSION = "profilers-v14-english"

# ---- model tiering ----------------------------------------------------------
# Not every dimension deserves the same model. `investment_logic` is the only one
# that performs cross-stream causal inference (attention -> research -> decision ->
# capital action) across four sources, so it keeps the strongest model. The others
# largely verbalise numbers the deterministic layer already computed, and a smaller
# model does that at a fraction of the cost.
#
# Tier resolution per dimension, first match wins:
#   1. LUNA_MODEL_<DIM>          explicit override for one dimension
#   2. DEEP mode                 LUNA_MODEL_DEEP (or strong) for every dimension
#   3. the dimension's tier      STRONG for logic, SMALL for the rest
STRONG_DIMS = {"logic"}
SMALL_MODEL = os.environ.get("LUNA_MODEL_SMALL", MODEL_VERSION)
STRONG_MODEL = os.environ.get("LUNA_MODEL_STRONG", MODEL_VERSION)
DEEP_MODEL = os.environ.get("LUNA_MODEL_DEEP", STRONG_MODEL)

# The model spends output budget on reasoning before emitting JSON. With the
# richer evidence pack a small cap truncates the completion and the structured
# parse fails, which would silently degrade every profile to the rule fallback.
MAX_TOKENS = int(os.environ.get("LUNA_MAX_TOKENS", "4000"))
# a smaller model needs less headroom; keep one knob but allow a separate cap
SMALL_MAX_TOKENS = int(os.environ.get("LUNA_MAX_TOKENS_SMALL", str(MAX_TOKENS)))


def resolve_model(dim, mode=None):
    """Model for a dimension, honouring per-dimension overrides and DEEP tiering."""
    explicit = os.environ.get(f"LUNA_MODEL_{str(dim).upper()}")
    if explicit:
        return explicit
    if mode == "deep":
        return DEEP_MODEL
    return STRONG_MODEL if dim in STRONG_DIMS else SMALL_MODEL

# ---------- schemas ----------

class RiskLabel(BaseModel):
    revealed_risk_behavior: Literal["conservative", "moderate", "aggressive", "very_aggressive", "unknown"]
    # v13: the risk level alone says how much risk this user carries. These say HOW
    # they carry it, which is what actually predicts the next move: a user who
    # averages down and one who cuts losses can both read "aggressive" and behave
    # oppositely in a drawdown.
    loss_response_style: Literal["averages_down", "holds", "cuts_losses", "unknown"] = "unknown"
    position_sizing: Literal["concentrated", "moderate", "diversified", "unknown"] = "unknown"
    leverage_usage: Literal["none", "occasional", "substantial", "unknown"] = "unknown"
    risk_budget_stability: Literal["stable", "adaptive", "erratic", "unknown"] = "unknown"
    drawdown_behavior: Literal["adds_into_weakness", "holds", "de_risks", "exits", "unknown"] = "unknown"
    evidence_strength: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    key_signals: list[str] = []
    unknown_fields: list[str] = []
    narrative: str = ""


class NextAction(BaseModel):
    """A forecast of the user's OWN next capital action.

    Not a market view and not a recommendation. The deterministic engine
    (`app/action_forecast.py`) ranks the candidates from the user's own behaviour;
    the model names the target and explains the reasoning. It must stay inside what
    the engine scored — a forecast invented from prose cannot be checked.
    """
    primary_action: Literal["start_new_position", "add_to_existing", "trim_position",
                            "exit_position", "hold", "unknown"] = "unknown"
    targets: list[str] = []
    horizon_days: Optional[int] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    # Which of the engine's signals drive this, in plain words.
    basis: list[str] = []
    # The observable that would show the prediction was wrong. Required: a forecast
    # that cannot be falsified cannot be checked, and an unchecked forecast is a guess.
    falsifier: str = ""
    rationale: str = ""


class LogicLabel(BaseModel):
    likely_styles: list[str] = []
    primary_style: Optional[str] = None
    thematic_focus: list[str] = []
    # v13: how the reasoning runs, not just what it concludes.
    decision_speed: Literal["deliberate", "prompt", "impulsive", "unknown"] = "unknown"
    conviction_style: Literal["scales_in", "one_shot", "mixed", "unknown"] = "unknown"
    thesis_durability: Literal["persistent", "rotating", "unknown"] = "unknown"
    primary_signal_source: list[str] = []
    next_action: Optional[NextAction] = None
    evidence_strength: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    key_signals: list[str] = []
    unknown_fields: list[str] = []
    narrative: str = ""


class PreferenceLabel(BaseModel):
    trading_style: Literal["long_term_investor", "swing_trader", "active_trader", "high_frequency", "observer_only", "unknown"]
    horizon: Literal["short", "medium", "long", "mixed", "unknown"]
    stated_vs_revealed: Literal["aligned", "divergent", "unknown"]
    notes: Optional[str] = None
    evidence_strength: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    key_signals: list[str] = []
    unknown_fields: list[str] = []
    narrative: str = ""


class EmotionLabel(BaseModel):
    disposition_effect: Literal["strong", "mild", "not_observed", "unknown"]
    chasing_momentum: Literal["yes", "no", "unknown"]
    panic_patterns: Literal["yes", "no", "unknown"]
    notes: Optional[str] = None
    evidence_strength: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    key_signals: list[str] = []
    unknown_fields: list[str] = []
    narrative: str = ""


class AssessmentLabel(BaseModel):
    experience_level: Literal["beginner", "intermediate", "advanced", "professional", "unknown"]
    lifecycle_stage: Literal["student", "early_career", "mid_career", "established", "retiree", "unknown"]
    investment_status: Literal["active", "learning_active", "passive_learning", "dormant", "unknown"]
    knowledge_domains: list[str] = []
    notes: Optional[str] = None
    evidence_strength: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    key_signals: list[str] = []
    unknown_fields: list[str] = []
    narrative: str = ""


SCHEMAS = {"risk": RiskLabel, "logic": LogicLabel, "preference": PreferenceLabel,
           "emotion": EmotionLabel, "assessment": AssessmentLabel}


class Decision(BaseModel):
    action: Literal["resolve", "approve_with_flags", "reject"]
    note: str = ""

# ---------- data access policy (harness enforces what each agent sees) ----------
# risk deliberately has NO conversation / behavior_intent access: stated
# interests and attention must not be smuggled in as revealed risk behavior.

POLICY = {
    # risk keeps its attention boundary: attention_capital is DERIVED FROM BM433
    # behaviour, so giving it to risk would smuggle behaviour back in. Position
    # dynamics comes from BM453 account data, which risk is allowed to see.
    #
    # `stated` (users.base_profile_summary) is NOT granted to any dimension: it was
    # person-identifying demographic text, not investment evidence. Removing it
    # from the policy lands the change through the evidence fingerprint, so every
    # dimension that used to see it is recomputed — no manual version bump needed.
    "risk": ["trade", "trading_digest", "position", "asset",
             "position_dynamics_digest"],
    # `action_forecast_digest` was here, and it is deliberately gone. The forecast
    # states an answer in the label vocabulary, so showing it to `logic` made the
    # model restate it in the narrative it writes — and the narrative is
    # published, which let a predictor read the engine's answer out of its own
    # input. Measured on the 80-user cohort: 961 of 1760 published profiles named
    # a copyable action, 960 of them the engine's. NO dimension sees the forecast;
    # it is archived at sample level (`forecast_baseline`) as the non-LLM baseline.
    "logic": ["behavior", "behavior_intent", "trade", "trading_digest",
              "behavior_trade_match", "position", "asset", "conversation",
              "position_dynamics_digest", "attention_capital_digest"],
    "preference": ["behavior", "behavior_intent", "trade", "trading_digest",
                   "position", "asset", "conversation",
                   "position_dynamics_digest", "attention_capital_digest"],
    "emotion": ["behavior", "behavior_intent", "trade", "trading_digest",
                "position", "asset", "conversation",
                "position_dynamics_digest", "attention_capital_digest"],
    "assessment": ["behavior", "behavior_intent", "trade", "trading_digest",
                   "position", "asset", "conversation",
                   "position_dynamics_digest", "attention_capital_digest"],
}

SYSTEM = (
    "You are a financial user-profile analyst. Reason only from the supplied evidence "
    "package and never invent missing data. Keep observed behavior (revealed) separate "
    "from user statements (stated); when evidence is insufficient, return unknown and "
    "name the gap in unknown_fields. Each key_signal must cite a supplied field or value. "
    "Do not provide investment advice. Return only schema-conforming JSON. Every natural-"
    "language output value, including narrative, basis, falsifier, rationale, notes, "
    "key_signals, and unknown_fields, must be in English. The narrative must be one to "
    "three concise, adviser-facing sentences explaining the observation, its basis, and "
    "material unknowns. Use plain English, cite at most two key figures, and do not add "
    "advice, suitability conclusions, market views, or unsupported predictions."
)

DIM_HINT = {
    "preference": (
        "If behavior is available but trades are unavailable, use observer_only rather "
        "than unknown unless behavior is extremely sparse. Conversation data is stated "
        "interest, not trading behavior. Use research depth, attention concentration, "
        "position turnover, book stability, entry/exit style, invested ratio, and the "
        "attention-versus-capital relationship to infer trading_style and horizon."
    ),
    "risk": (
        "Without trade and account evidence, revealed_risk_behavior must be unknown. "
        "Use drawdown, volatility, short exposure, turnover, validated invested ratio, "
        "leverage, concentration, position changes, loss-closing behavior, and underwater "
        "adds. Treat withheld or thin-sample ratios as unknown. Browser attention is not "
        "risk evidence. Infer loss_response_style, position_sizing, leverage_usage, "
        "risk_budget_stability, and drawdown_behavior independently; never infer them "
        "from the overall risk label alone."
    ),
    "logic": (
        "Derive thematic_focus and likely_styles only from supplied behavior, trade, "
        "position, and conversation evidence. Use intent stages, research style, research-"
        "to-trade latency, attention-capital gaps, conversion, and persistence to describe "
        "the observed decision process without claiming motives. Leave next_action null "
        "and do not forecast the user's next capital action anywhere in your output, not "
        "in narrative or key_signals either: that forecast is produced deterministically "
        "outside this profile and is the experiment's baseline, so a restatement of it "
        "here would end up inside the evidence a predictor reads."
    ),
    "emotion": (
        "Infer disposition, momentum chasing, or panic only from temporal behavior and "
        "trade patterns. Respect sample notes. Negative unrealized days are stress "
        "exposure, not proof of panic. Conversation data has no timestamp and cannot "
        "establish these patterns. Loss-conditioned attention is correlation, not cause."
    ),
    "assessment": (
        "Infer experience only from observable breadth, research depth, instrument use, "
        "account history, turnover, and portfolio management. With insufficient evidence, "
        "return unknown. lifecycle_stage is unknown without direct evidence. active requires "
        "executed trades; browsing alone supports learning_active or passive_learning."
    ),
}


def _client(max_tokens=None, model=None):
    from langchain_openai import ChatOpenAI
    key = (os.environ.get("AIME_LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("LITELLM_API_KEY"))
    if not key:
        return None
    return ChatOpenAI(model=model or MODEL_VERSION, api_key=key,
                      base_url=os.environ.get("LUNA_BASE_URL",
                                              "https://api.openai.com/v1"),
                      temperature=0, max_tokens=max_tokens or MAX_TOKENS)


# ---------- deterministic fallbacks ----------

def _pct(v, digits=1):
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(v)


def _rule_narr(summary, facts, unknowns):
    """Compose a concise English rule-layer narrative."""
    parts = [summary.rstrip(".") + "."]
    if facts:
        parts.append("Observed evidence: " + "; ".join(facts[:3]) + ".")
    else:
        parts.append("The available data does not support a reliable conclusion.")
    if unknowns:
        parts.append("Still unknown: " + "; ".join(unknowns[:3]) + ".")
    return " ".join(parts)


def rule_risk(ev):
    t, a, p = ev["trade"], ev["asset"], ev["position"]
    # avg invested ratio is owned (and validated) by position_dynamics; the copy
    # in `asset` was removed. This rule used to read the removed field, which
    # would have silently stopped firing — a dead rule degrades risk detection
    # without any error surfacing.
    cm = ((ev.get("position_dynamics_digest") or {}).get("capital_management") or {})
    inv = cm.get("avg_invested_ratio")
    signals, unknowns = [], []
    score = 0
    if t.get("available"):
        if t["trade_count"] >= 20:
            score += 2; signals.append(f"trade.trade_count={t['trade_count']}")
        if t.get("leveraged_etf_trade_share", 0) >= 0.1:
            score += 2; signals.append(f"trade.leveraged_etf_trade_share={t['leveraged_etf_trade_share']}")
    if a.get("available"):
        if (a.get("max_drawdown") or 0) <= -0.3:
            score += 1; signals.append(f"asset.max_drawdown={a.get('max_drawdown')}")
        if inv is not None and inv >= 0.9:
            score += 1
            signals.append(f"position_dynamics_digest.capital_management.avg_invested_ratio={inv}")
        if (a.get("equity_volatility") or 0) >= 0.03:
            score += 1; signals.append(f"asset.equity_volatility={a.get('equity_volatility')}")
    if p.get("available"):
        if p.get("days_with_shorts", 0) >= 2:
            score += 1; signals.append(f"position.days_with_shorts={p['days_with_shorts']}")
        if (p.get("holding_turnover_rate") or 0) >= 0.5:
            score += 2; signals.append(f"position.holding_turnover_rate={p.get('holding_turnover_rate')}")
        if (p.get("avg_top1_concentration") or 0) >= 0.5:
            score += 1; signals.append(f"position.avg_top1_concentration={p.get('avg_top1_concentration')}")
    if not signals:
        unknowns = ["No trade, position, or account evidence is available"]
        return RiskLabel(revealed_risk_behavior="unknown", evidence_strength="low",
                         confidence=0.2, key_signals=[], unknown_fields=unknowns,
                         narrative=_rule_narr("There is insufficient trade and account evidence to assess revealed risk behavior",
                                              facts=[], unknowns=unknowns)).model_dump()
    label = "very_aggressive" if score >= 6 else "aggressive" if score >= 4 else "moderate" if score >= 2 else "conservative"
    strength = "high" if score >= 5 and len(signals) >= 3 else "medium" if score >= 2 else "low"
    zh = {"very_aggressive": "very aggressive", "aggressive": "aggressive", "moderate": "moderate", "conservative": "conservative"}[label]

    facts = []
    if t.get("available"):
        if t["trade_count"] >= 20:
            facts.append(f"{t['trade_count']} executed trades indicate frequent activity")
        if t.get("leveraged_etf_trade_share", 0) >= 0.1:
            facts.append(f"{_pct(t['leveraged_etf_trade_share'], 0)} of trades involved leveraged ETFs")
    if a.get("available"):
        if (a.get("max_drawdown") or 0) <= -0.3:
            facts.append(f"the account experienced a {_pct(a['max_drawdown'])} maximum drawdown")
        if inv is not None and inv >= 0.9:
            facts.append(f"an average {_pct(inv, 0)} of assets was invested in equities")
    if p.get("available"):
        if p.get("days_with_shorts", 0) >= 2:
            facts.append(f"short positions appeared on {p['days_with_shorts']} days")
        if (p.get("holding_turnover_rate") or 0) >= 0.5:
            facts.append(f"holding turnover reached {_pct(p['holding_turnover_rate'])}")
        if (p.get("avg_top1_concentration") or 0) >= 0.5:
            facts.append("the book was concentrated in a small number of securities")

    ru = []
    if not t.get("available"):
        ru.append("Risk capacity has not been verified through a suitability assessment")
    return RiskLabel(revealed_risk_behavior=label, evidence_strength=strength,
                     confidence=min(0.55 + 0.06 * len(signals), 0.85),
                     key_signals=signals, unknown_fields=unknowns,
                     narrative=_rule_narr(f"Observed trade and account behavior is {label}",
                                          facts, ru)).model_dump()


def rule_logic(ev):
    b, t, m = ev["behavior"], ev["trade"], ev["behavior_trade_match"]
    styles, focus, signals = [], [], []
    if t.get("available") and t["trade_count"]:
        styles.append("event_driven_trading")
        signals.append(f"trade.top_tickers={[x['ticker'] for x in t['top_tickers'][:3]]}")
    if b.get("available"):
        themes = {}
        for x in b.get("top_viewed_tickers", []):
            themes[x["theme"]] = themes.get(x["theme"], 0) + x["count"]
        top_theme = max(themes, key=themes.get) if themes else None
        if top_theme and top_theme != "Other":
            focus.append(top_theme); signals.append(f"behavior.top_viewed_tickers theme={top_theme}")
    if m.get("available") and m["prior_view_rate"] >= 0.3:
        styles.append("research_before_trade")
        signals.append(f"behavior_trade_match.prior_view_rate={m['prior_view_rate']}")
    cv = ev.get("conversation") or {}
    if cv.get("available"):
        cthemes = [x["theme"] for x in (cv.get("themes") or []) if x.get("theme") != "Other"]
        for th in cthemes[:2]:
            if th not in focus:
                focus.append(th)
        if cthemes:
            signals.append(f"conversation.themes={cthemes[:3]}")
    if not styles:
        styles = ["insufficient_data"]
    lfacts = []
    if focus:
        lfacts.append("Sustained attention centered on " + ", ".join(focus[:3]))
    if t.get("available") and t.get("top_tickers"):
        lfacts.append("Trading concentrated in " + ", ".join(x["ticker"] for x in t["top_tickers"][:3]))
    if m.get("available") and m.get("prior_view_rate", 0) >= 0.3:
        lfacts.append(f"{_pct(m['prior_view_rate'], 0)} of trades had prior research activity")
    _lru = [] if signals else ["No identifiable relationship between research behavior and trading is established"]
    zh_style = {"event_driven_trading": "event-driven", "research_before_trade": "research-led",
                "insufficient_data": "unclear"}
    return LogicLabel(likely_styles=styles, primary_style=styles[0], thematic_focus=focus,
                      evidence_strength="medium" if signals else "low",
                      confidence=0.5 if signals else 0.2,
                      key_signals=signals,
                      unknown_fields=_lru,
                      narrative=_rule_narr(
                          f"The observed relationship among browsing, holdings, and trades is most consistent with {styles[0]}",
                          lfacts, _lru)).model_dump()


def rule_preference(ev):
    b, t, p = ev["behavior"], ev["trade"], ev["position"]
    signals, unknowns = [], []
    if t.get("available") and t["trade_count"]:
        if t["max_trades_per_day"] >= 5 or (t.get("median_gap_between_trading_days") or 99) <= 1:
            style = "high_frequency" if t["max_trades_per_day"] >= 20 else "active_trader"
        else:
            style = "swing_trader"
        signals.append(f"trade.max_trades_per_day={t['max_trades_per_day']}")
    elif b.get("available") and b["event_count"]:
        style = "observer_only"
        signals.append(f"behavior.event_count={b['event_count']}, trade.available=False")
    else:
        style = "unknown"
        unknowns.append("Neither browsing behavior nor executed trades are available")
    if p.get("available") and p.get("days") and (p.get("max_num_positions") or 0) > 0:
        stab, turn = p.get("position_count_stability"), p.get("holding_turnover_rate")
        if (turn is not None and turn <= 0.2) or (stab or 0) >= 0.8:
            horizon = "long" if style in ("long_term_investor", "swing_trader", "observer_only") else "mixed"
            signals.append(f"position.holding_turnover_rate={turn}, position.position_count_stability={stab}")
        elif turn is not None and turn >= 0.5:
            horizon = "short"
            signals.append(f"position.holding_turnover_rate={turn}")
        else:
            horizon = "medium"
    else:
        horizon = "unknown"; unknowns.append("No observable position book is available")
    cv = ev.get("conversation") or {}
    if cv.get("available") and cv.get("self_reference_share", 0) >= 0.3:
        signals.append(f"conversation.self_reference_share={cv.get('self_reference_share')}")
    # `stated_vs_revealed` used to be reported as `aligned` whenever the removed
    # self-report text happened to be non-empty AND any behavioural signal existed.
    # That was never a comparison of the two — it could not disagree with anything —
    # so it is not replaced with another guess. The rule layer stays conservative and
    # the LLM decides from the conversation evidence it actually has; `unknown` is the
    # honest answer when there is no stated side to compare against.
    svr = "unknown"
    conf = 0.6 if signals else 0.2
    zh_style = {"high_frequency": "high-frequency", "active_trader": "active trading", "swing_trader": "swing trading",
                "observer_only": "observation only", "unknown": "unclear"}[style]
    zh_horizon = {"short": "short", "medium": "medium", "long": "long", "mixed": "mixed",
                  "unknown": "unclear"}[horizon]
    pfacts = []
    if t.get("available") and t.get("trade_count"):
        if t.get("max_trades_per_day", 0) >= 5:
            pfacts.append(f"up to {t['max_trades_per_day']} orders in one day indicates a fast trading cadence")
        else:
            pfacts.append(f"{t['trade_count']} executed trades indicate a measured cadence")
    elif b.get("available"):
        pfacts.append(f"{b.get('event_count', 0)} browsing events but no executed trades indicate observation-only activity")
    if p.get("available") and p.get("max_num_positions"):
        if (p.get("holding_turnover_rate") or 0) <= 0.2:
            pfacts.append(f"holding turnover of {_pct(p['holding_turnover_rate'])} indicates relatively persistent positions")
        elif (p.get("holding_turnover_rate") or 0) >= 0.5:
            pfacts.append(f"holding turnover of {_pct(p['holding_turnover_rate'])} indicates a shorter horizon")
    return PreferenceLabel(trading_style=style, horizon=horizon, stated_vs_revealed=svr,
                           notes=None, evidence_strength="medium" if len(signals) >= 2 else "low",
                           confidence=conf, key_signals=signals, unknown_fields=unknowns,
                           narrative=_rule_narr(
                               f"Observed trading style is {style}, with a {horizon} horizon",
                               pfacts, unknowns)).model_dump()


def rule_emotion(ev):
    b, t, p = ev["behavior"], ev["trade"], ev["position"]
    a = ev.get("asset") or {}
    signals, unknowns = [], []
    chasing = panic = "unknown"
    disposition = "unknown"
    if b.get("available") and (b.get("short_timeframe_share") or 0) >= 0.5:
        chasing = "yes"; signals.append(f"behavior.short_timeframe_share={b.get('short_timeframe_share')}")
    loss_days = (p or {}).get("days_with_negative_unrealized_profit", 0)
    turn = (p or {}).get("holding_turnover_rate")
    if loss_days >= 5:
        signals.append(f"position.days_with_negative_unrealized_profit={loss_days}")
        if turn is not None and turn >= 0.5:
            panic = "yes"
            signals.append(f"position.holding_turnover_rate={turn}")
    if a.get("available") and (a.get("max_drawdown") or 0) <= -0.3:
        signals.append(f"asset.max_drawdown={a.get('max_drawdown')}")
    if not t.get("available") or (t.get("trade_count", 0) < 5):
        disposition = "unknown"; unknowns.append("No usable profit/loss and close-timing series is available")
    if not signals:
        unknowns.append("Observable emotion-related behavioral signals are insufficient")
    efacts = []
    if b.get("available") and (b.get("short_timeframe_share") or 0) >= 0.5:
        efacts.append(f"{_pct(b['short_timeframe_share'], 0)} of chart views used short timeframes")
    if loss_days >= 5:
        efacts.append(f"the account remained invested through {loss_days} days with unrealized losses")
    if a.get("available") and (a.get("max_drawdown") or 0) <= -0.3:
        efacts.append(f"the account experienced a {_pct(a['max_drawdown'])} maximum drawdown")
    erru = list(unknowns)
    if not t.get("available") or (t.get("trade_count", 0) < 5):
        erru.append("The trade sample is too small to assess selling behavior under gains or losses")
    return EmotionLabel(disposition_effect=disposition, chasing_momentum=chasing,
                        panic_patterns=panic, notes=None,
                        evidence_strength="medium" if len(signals) >= 2 else "low",
                        confidence=0.5 if signals else 0.2,
                        key_signals=signals, unknown_fields=unknowns,
                        narrative=_rule_narr(
                            "Emotion-related tendencies are only indirectly observable through behavior timing and account variation; no direct psychological evidence is available",
                            efacts, erru)).model_dump()


def rule_assessment(ev):
    # `ev["stated"]` was indexed directly; the block no longer exists (it carried the
    # person-identifying self-report channel). `.get` keeps this rule path working for
    # users whose saved profile predates the removal.
    b, t, s = ev["behavior"], ev["trade"], ev.get("stated") or {}
    signals, unknowns = [], []
    # The self-report keyword branch that read "less than a year" / "beginner" from
    # `stated_background` is gone with the channel. It was the only path that made the
    # rule layer claim `beginner`, and it derived experience from a demographic blob
    # rather than from observed behaviour — so experience now comes from behaviour
    # (revealed) or falls to `unknown`, which is the more defensible order anyway.
    if b.get("available") and b.get("option_views", 0) > 0:
        exp = "intermediate"; signals.append(f"behavior.option_views={b.get('option_views')}")
    elif b.get("available") and b["event_count"]:
        exp = "unknown"; unknowns.append("No stated background is available and the behavioral evidence is not direct enough")
    else:
        exp = "unknown"; unknowns.append("No usable experience-related evidence is available")
    n_tk = (t.get("unique_tickers") or 0) + (b.get("tickers_viewed") or 0)
    status = ("active" if t.get("available") else
              "learning_active" if (b.get("available") and b["event_count"] >= 100) else
              "passive_learning" if b.get("available") else "unknown")
    if status == "active":
        signals.append(f"trade.trade_count={t.get('trade_count')}")
    elif status == "learning_active":
        signals.append(f"behavior.event_count={b['event_count']}, trade.available=False")
    domains = []
    if b.get("available"):
        themes = {}
        for x in b.get("top_viewed_tickers", []):
            themes[x["theme"]] = themes.get(x["theme"], 0) + x["count"]
        domains = [k for k, c in sorted(themes.items(), key=lambda kv: -kv[1])[:3] if k != "Other"]
        if domains:
            signals.append(f"behavior.top_viewed_tickers themes={domains}")
    cv = ev.get("conversation") or {}
    if cv.get("available"):
        cthemes = [x["theme"] for x in (cv.get("themes") or []) if x.get("theme") != "Other"]
        for th in cthemes[:3]:
            if th not in domains:
                domains.append(th)
        if cthemes:
            signals.append(f"conversation.themes={cthemes[:3]}")
        if cv.get("self_reference_share", 0) >= 0.3:
            signals.append(f"conversation.self_reference_share={cv.get('self_reference_share')}")
    unknowns.append("Life stage cannot be inferred from behavioral data")
    zh_exp = {"beginner": "beginner", "intermediate": "intermediate", "advanced": "advanced",
              "professional": "professional", "unknown": "unclear"}[exp]
    zh_status = {"active": "executed trades indicate active participation", "learning_active": "active learning and observation",
                 "passive_learning": "occasional low-intensity observation", "dormant": "little recent activity",
                 "unknown": "activity level is unknown"}[status]
    afacts = []
    # The `if stated:` branch that used to prepend "用户自述交易经验不足一年、仍在学习"
    # is gone with the channel: it reported a self-report we no longer ingest, so the
    # claim would have been unsupported. Observed status carries the statement.
    afacts.append(zh_status)
    if domains:
        afacts.append("Observed knowledge domains include " + ", ".join(domains[:3]))
    if b.get("available") and b.get("tickers_viewed"):
        afacts.append(f"research activity covered {b['tickers_viewed']} distinct securities")
    return AssessmentLabel(experience_level=exp, lifecycle_stage="unknown",
                           investment_status=status, knowledge_domains=domains, notes=None,
                           evidence_strength="medium" if len(signals) >= 2 else "low",
                           confidence=0.55 if signals else 0.2,
                           key_signals=signals, unknown_fields=unknowns,
                           narrative=_rule_narr(
                               f"Observed investment experience is {exp}", afacts,
                               ["life stage", "stated investment background"])).model_dump()


RULES = {"risk": rule_risk, "logic": rule_logic, "preference": rule_preference,
         "emotion": rule_emotion, "assessment": rule_assessment}

# ---------- LLM agent runner ----------

_WS_RE = re.compile(r"\s+")
NARRATIVE_MAX_CHARS = 600


def _clean_narrative(text, fallback_text=""):
    """Normalise LLM prose: collapse whitespace, cap length, never leave empty.

    A blank or runaway narrative is the one thing a human-facing field must not
    be, so an empty result falls back to the rule-layer text.
    """
    s = _WS_RE.sub(" ", str(text or "")).strip()
    if len(s) > NARRATIVE_MAX_CHARS:
        s = s[:NARRATIVE_MAX_CHARS].rstrip() + "…"
    return s or fallback_text


def run_profiler(dimension, evidence, timeout_note="", mode=None, fallback_evidence=None):
    """Return (output_dict, source).

    source is `llm`, `rule`, or a *reasoned* fallback such as
    `rule:llm_error:LengthFinishReasonError` — a silent downgrade to the rule
    layer must never be indistinguishable from a real LLM run.

    `evidence` is the token-projected view the model will see. `fallback_evidence`
    is the policy-filtered FULL evidence for the deterministic rule layer, which
    is not token-constrained and reads whole blocks (e.g. `ev["trade"]`); passing
    the projected view there raised KeyError on every rule fallback.

    `mode` selects the model tier (DEEP promotes every dimension to the strong
    model). The chosen model is recorded in the output so a weaker narrative can
    be traced to the tier that produced it rather than looking like a regression.
    """
    rule_input = fallback_evidence if fallback_evidence is not None else evidence
    fallback = RULES[dimension](rule_input)
    if dimension not in POLICY:
        raise ValueError(f"unknown profiler: {dimension}")
    # `evidence` arrives already permission-filtered by orchestrator.agent_view;
    # the filter is repeated here so a direct caller cannot bypass it.
    permitted = {k: evidence[k] for k in POLICY[dimension] if evidence.get(k, {}).get("available")}
    model = resolve_model(dimension, mode)
    messages = [
        ("system", SYSTEM + "\n" + DIM_HINT.get(dimension, "")),
        ("user", json.dumps({"dimension": dimension, "evidence": permitted}, ensure_ascii=False) + timeout_note),
    ]

    def _attempt(budget):
        client = _client(budget, model=model)
        if client is None:
            return None
        # include_raw so the provider's token usage is visible
        llm = client.with_structured_output(SCHEMAS[dimension], include_raw=True)
        res = llm.invoke(messages)
        from .token_meter import record
        record("dimension", model, getattr(res.get("raw"), "usage_metadata", None))
        out = res.get("parsed")
        if out is None:
            return None
        d = out.model_dump()
        # every key_signal must literally reference a permitted evidence field
        blob = json.dumps(permitted, ensure_ascii=False)
        d["key_signals"] = [s for s in d.get("key_signals", []) if re.search(r"[a-zA-Z_]", s) and any(
            tok in blob for tok in re.findall(r"[A-Za-z_]{3,}", s))] or fallback["key_signals"]
        d["narrative"] = _clean_narrative(d.get("narrative"), fallback.get("narrative", ""))
        d["model"] = model
        return d

    # a small model needs less reasoning headroom than the strong one
    base_budget = SMALL_MAX_TOKENS if model == SMALL_MODEL else MAX_TOKENS
    try:
        d = _attempt(base_budget)
        if d is None:
            fallback["narrative"] = _clean_narrative(fallback.get("narrative"))
            fallback["model"] = None
            return fallback, "rule"
        return d, f"llm:{model}"
    except Exception as e:
        name = type(e).__name__
        # reasoning ate the whole budget: retry once with double the headroom
        if "Length" in name or "length limit" in str(e).lower():
            try:
                d = _attempt(base_budget * 2)
                if d is not None:
                    return d, f"llm:{model}"
                fallback["narrative"] = _clean_narrative(fallback.get("narrative"))
                fallback["model"] = None
                return fallback, "rule:llm_length"
            except Exception as e2:
                fallback["narrative"] = _clean_narrative(fallback.get("narrative"))
                fallback["model"] = None
                return fallback, f"rule:llm_error:{type(e2).__name__}"
        fallback["narrative"] = _clean_narrative(fallback.get("narrative"))
        return fallback, f"rule:llm_error:{name}"
