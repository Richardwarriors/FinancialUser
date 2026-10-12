"""Curated L2 taxonomy: turn the deterministic engines into 10 profiles worth of depth.

Why this module exists
---------------------
The tree had the right 10 L1 and almost nothing under them: 6 of the 10 had ZERO L2,
because their L2s were sourced from the KYC tree and the KYC features are mostly
`unavailable`. What shipped looked like this:

    goals                       L2=0
    constraints                 L2=0
    risk_preference             L2=0
    investment_logic            L2=0
    personalization_and_rules   L2=0
    investor_assessment         L2=0

That is a table of contents, not a profile. An L1 with no subdivision cannot be acted
on: 「风险偏好: 激进」 and 「风险偏好 › 结构性敞口 › 杠杆占比 42%」 support completely
different decisions.

This module builds the missing depth from the evidence that IS available — behavior_events
(behaviour), trade_executions (trades), account_snapshots (daily account) and the deterministic engines
derived from them — rather than from the questionnaire fields that never arrive.

Design rules, inherited from the rest of the tree:

* **Every leaf is computed, not composed.** No LLM here. A leaf exists only when the
  field behind it exists, and it carries the citation that produced it.
* **Omit rather than zero-fill.** A user with no trades gets no 「胜率 0%」; the L2 is
  simply absent and the absence is counted.
* **Separate the four risk questions.** Willingness (what they say), capacity (what
  they can bear), exposure (what they actually hold) and response (what they do in a
  drawdown) are different facts and must never be merged into one label.
* **A proxy says it is a proxy**, with the basis stated and confidence capped.

The output shape matches what `profile_tree` already consumes:
`{l1_key: {l2_key: {"label", "hint", "leaves": [leaf, ...]}}}`.
"""
from .metric_glossary import window_for

L2_VERSION = "l2-taxonomy-v1"

# Re-exported by the caller; kept local so this module stands alone.
FILLED, PROXY, GAP = "filled", "proxy", "gap"

# A rate needs a real denominator. Same discipline as everywhere else.
MIN_TRADES_FOR_RATIO = 3
MIN_DAYS_FOR_STYLE = 20
# A proxy may not claim confidence above this: it is an inference from behaviour,
# not a measured answer to the question.
PROXY_CONF_CAP = 0.6


def _cnt(v):
    """Number of items in a container, or the value itself when it is a scalar.

    `asked_about_own_portfolio` is a LIST of conversation titles while its sibling
    `self_reference_count` is an INT. Interpolating the list into an f-string put the
    whole Python repr into a prose field — the exact repr leak the tree guards scan
    for, caught by `test_no_repr_leak` on the very next run. Never `len()`-guess:
    handle both shapes explicitly.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple, set, dict)):
        return len(v)
    return v


def _n(x, digits=1):
    if x is None:
        return "—"
    try:
        return f"{float(x):,.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x, digits=1):
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(x)


def _ratio(num, den):
    if num is None or not den:
        return None
    return num / den


def _clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# The profile document is keyed by LONG dimension names (`investor_assessment`,
# `emotion_and_psychology`, `preferences`) while POLICY, the projection and these
# builders use SHORT ones (`assessment`, `emotion`, `preference`). Looking up the
# short name on a profile returns {} — silently, no error — so every leaf that cites
# a dimension vanished. This exact mismatch is documented in the project notes as
# having caused a previous failure; it recurred here.
_DIM_ALIASES = {
    "risk": "risk", "logic": "investment_logic",
    "preference": "preferences", "emotion": "emotion_and_psychology",
    "assessment": "investor_assessment",
}


def _dim(dims, short):
    """Read a dimension by its SHORT name from a profile keyed by LONG names."""
    dims = dims or {}
    if short in dims and isinstance(dims[short], dict):
        return dims[short]
    return dims.get(_DIM_ALIASES.get(short, short)) or {}


def leaf(key, label, display, status=FILLED, value=None, evidence=None,
         confidence=None, proxy_basis=None, caveat=None):
    """One L3 fact, in the shape `profile_tree._leaf` expects.

    `display` must already be prose — never `str(value)` of a container.
    """
    return {"key": key, "label": label, "display": display, "status": status,
            "value": value if value is not None else display,
            "evidence": [e for e in (evidence or []) if e],
            "confidence": confidence,
            "proxy_basis": proxy_basis, "caveat": caveat,
            "data_needed": []}


def _l2(label, leaves, hint=None):
    """An L2 block. Empty leaf lists are dropped by the caller, never shown."""
    return {"label": label, "hint": hint, "leaves": [x for x in leaves if x]}


def _top(seq, label_key, value_key=None, limit=4):
    """`[{k:..,v:..}]` -> "名称 值、…", shape-tolerantly.

    The engines are inconsistent by design here: some publish a LIST of
    `{"info_type", "count", "share"}` records, others a DICT of name -> count.
    Assuming one shape is how a whole L1 silently vanished — `_background` threw
    `AttributeError: 'list' object has no attribute 'items'`, the builder's error
    guard caught it, and the L1 shipped EMPTY rather than wrong. Empty is the safer
    failure, but it is still a failure, so both shapes are handled explicitly.
    """
    rows = []
    if isinstance(seq, dict):
        rows = [(str(k), v) for k, v in seq.items()]
    elif isinstance(seq, list):
        for x in seq:
            if isinstance(x, dict):
                name = x.get(label_key) or x.get("name") or x.get("ticker")
                val = x.get(value_key) if value_key else None
                if val is None:
                    val = x.get("count", x.get("share", x.get("value")))
                if name is not None:
                    rows.append((str(name), val))
            elif x is not None:
                rows.append((str(x), None))
    def _key(kv):
        v = kv[1]
        if isinstance(v, dict):
            v = v.get("share", v.get("count", 0))
        try:
            return -float(v)
        except (TypeError, ValueError):
            return 0.0
    rows.sort(key=_key)
    out = []
    for name, val in rows[:limit]:
        if isinstance(val, float):
            out.append(f"{name} {_pct(val)}")
        elif val is None:
            out.append(name)
        else:
            out.append(f"{name} {val}")
    return "、".join(out)


# ---------------------------------------------------------------------------
# 目标 — what the user is trying to achieve, inferred from how they act
# ---------------------------------------------------------------------------
def _goals(ev):
    bi, ac = ev.get("behavior_intent") or {}, ev.get("attention_capital") or {}
    pd = ev.get("position_dynamics") or {}
    pm, tm = ev.get("portfolio_metrics") or {}, ev.get("trade_metrics") or {}
    asset = ev.get("asset") or {}
    style = pd.get("style") or {}
    cap = pd.get("capital_management") or {}
    lt = pm.get("long_term") or {}
    ex = tm.get("execution") or {}
    res = tm.get("results") or {}

    out = {}

    # --- 时间维度的目标 -----------------------------------------------------
    leaves = []
    avg_hold = ex.get("median_hold_days")
    horizon = (ev.get("dimensions") or {}).get("preference", {}).get("horizon")
    if avg_hold is not None:
        leaves.append(leaf(
            "holding_period_objective", "Holding period target (observable)",
            f"Median holding period of closed-trade positions: {_n(avg_hold)} days"
            + (", classified as long-term holding" if avg_hold >= 60 else
               ", falls under multi-week swing trading" if avg_hold >= 10 else ", falls under short-term trading"),
            evidence=[f"trade_metrics.execution.median_hold_days={avg_hold}",
                      f"trade_metrics.execution.matched_round_trips="
                      f"{ex.get('matched_round_trips')}"],
            confidence=0.8))
    hp = (lt.get("holding_endurance") or {}).get("avg_holding_weeks")
    if hp is not None:
        leaves.append(leaf(
            "position_holding_objective", "Position holding period target (full window)",
            f"Current positions held for an average of {_n(hp)} weeks",
            evidence=[f"portfolio_metrics.long_term.holding_endurance"
                      f".avg_holding_weeks={hp}"], confidence=0.7))
    turn = (pm.get("short_term") or {}).get("short_term_trading", {}).get("turnover_7d")
    if turn is not None:
        leaves.append(leaf(
            "turnover_objective", "Turnover rhythm",
            f"Turnover over the past 7 days {_pct(turn)}"
            + (", trades frequently" if turn > 0.5 else ", trading activity is moderate" if turn > 0.1 else ", makes almost no moves"),
            evidence=[f"portfolio_metrics.short_term.short_term_trading"
                      f".turnover_7d={turn}"], confidence=0.8))
    if leaves:
        out["time_horizon"] = _l2("Time-based objectives", leaves,
                                  "Inferred from actual holding periods and turnover pace, not a questionnaire answer.")

    # --- 收益目标 -----------------------------------------------------------
    leaves = []
    if res.get("closed_trades") is not None:
        n = res["closed_trades"]
        leaves.append(leaf(
            "realised_result_basis", "Realized return basis",
            f"{n} trades closed, total realized P&L {_n(res.get('total_realised_pnl'), 0)}"
            + (f"(Win rate {_pct(res.get('win_rate'))})"
               if res.get("win_rate") is not None else ""),
            evidence=[f"trade_metrics.results.closed_trades={n}",
                      f"trade_metrics.results.total_realised_pnl="
                      f"{res.get('total_realised_pnl')}"],
            confidence=0.8,
            caveat=("When there are fewer than 8 samples, skewness/kurtosis is not provided, and the win rate itself fluctuates greatly"
                    if n < 8 else None)))
    ret = asset.get("equity_change_pct")
    if ret is not None and asset.get("return_reliable"):
        leaves.append(leaf(
            "return_objective_realised", "Account return (time-weighted)",
            f"Equity change during the observation window: {_pct(ret)}, maximum drawdown: {_pct(asset.get('max_drawdown'))}"
            + (f"（{asset.get('max_drawdown_date')}）" if asset.get("max_drawdown_date") else ""),
            evidence=["asset.equity_change_pct", "asset.max_drawdown",
                      "asset.flow_adjusted"],
            confidence=0.85,
            caveat="Deposits and withdrawals excluded; includes the unobservable portion of external assets."))
    if leaves:
        out["return_objective"] = _l2("Return targets and realization", leaves)

    # --- 资金目标：增值 vs 保值 ---------------------------------------------
    leaves = []
    inv = cap.get("avg_invested_ratio")
    if inv is not None:
        leaves.append(leaf(
            "capital_deployment_objective", "Capital investment inclination",
            f"Average position share {_pct(inv)}"
            + (", close to fully invested, with a preference for appreciation" if inv >= 0.8 else
               ", retaining some cash for flexibility" if inv >= 0.4 else
               ", mainly holds cash, with goals focused on capital preservation and timing"),
            evidence=[f"position_dynamics.capital_management.avg_invested_ratio={inv}",
                      f"position_dynamics.capital_management.coverage="
                      f"{cap.get('coverage')}"],
            confidence=0.85))
    cash = cap.get("cash_buffer_ratio")
    if cash is not None:
        leaves.append(leaf(
            "cash_target", "Cash buffer",
            f"Average cash share {_pct(cash)}",
            evidence=[f"position_dynamics.capital_management.cash_buffer_ratio={cash}"],
            confidence=0.8))
    band = ((ev.get("portfolio_metrics") or {}).get("long_term") or {}).get(
        "investable_assets_band") or {}
    if band.get("band"):
        leaves.append(leaf(
            "investable_band", "Investable amount range",
            f"Estimated investable amount range based on account balance: {band['band']}",
            status=PROXY, proxy_basis="Infer from the account balance and cash ratio, not questionnaire responses",
            confidence=min(PROXY_CONF_CAP, 0.55),
            evidence=["computed_portfolio_metrics.investable_assets_band"]))
    if leaves:
        out["capital_growth_vs_preservation"] = _l2("Preference for appreciation and capital preservation", leaves)

    # --- 目标一致性 ---------------------------------------------------------
    leaves = []
    conv = (ac.get("conversion") or {})
    c_rate = conv.get("research_to_holding_conversion")
    if c_rate is not None and conv.get("research_to_holding_denominator"):
        leaves.append(leaf(
            "intent_to_action_gap", "Gap between intent and action",
            f"{conv.get('research_to_holding_denominator')} instruments monitored but not held; only {conv.get('research_to_holding_numerator')} were ultimately purchased ({_pct(c_rate)})"
            + (", indicating that research exceeds execution and the target may remain at the observation stage"
               if c_rate < 0.1 else ", research and execution are broadly connected"),
            evidence=["attention_capital.conversion."
                      "research_to_holding_conversion",
                      "attention_capital.conversion."
                      "research_to_holding_denominator"],
            confidence=0.7))
    if style.get("accumulation_episodes") is not None:
        acc, dist = style["accumulation_episodes"], style.get("distribution_episodes") or 0
        leaves.append(leaf(
            "objective_direction", "Direction of capital movements",
            f"{acc} add-position episodes vs {dist} trim-position episodes,"
            + ("Still in the net position-building phase" if acc > dist else
               "In a net trimming phase" if dist > acc else "Add/trim balance"),
            evidence=["position_dynamics.style.accumulation_episodes",
                      "position_dynamics.style.distribution_episodes"],
            confidence=0.75))
    if leaves:
        out["objective_consistency"] = _l2("Objective consistency", leaves)

    return out


# ---------------------------------------------------------------------------
# 约束 — what limits the user, inferred rather than asked
# ---------------------------------------------------------------------------
def _constraints(ev):
    pd = ev.get("position_dynamics") or {}
    pm, tm = ev.get("portfolio_metrics") or {}, ev.get("trade_metrics") or {}
    ac = ev.get("attention_capital") or {}
    asset = ev.get("asset") or {}
    cap = pd.get("capital_management") or {}
    conv = pd.get("conviction") or {}
    ex = tm.get("execution") or {}
    act = tm.get("activity") or {}
    tdig = (ev.get("trading_digest") or {}).get("portfolio") or {}

    out = {}

    # --- 资金规模 -----------------------------------------------------------
    leaves = []
    mv = (pm.get("short_term") or {}).get("portfolio_snapshot", {}).get("total_market_value")
    if mv is not None:
        leaves.append(leaf(
            "capital_scale", "Account capital scale",
            f"Current securities market value {_n(mv, 0)}",
            evidence=[f"portfolio_metrics.short_term.portfolio_snapshot"
                      f".total_market_value={mv}"], confidence=0.9))
    ts = ex.get("avg_trade_size")
    if ts is not None:
        leaves.append(leaf(
            "trade_size_constraint", "Order size per transaction",
            f"Average per trade {_n(ts, 0)}, median {_n(ex.get('median_trade_size'), 0)}",
            evidence=["trade_metrics.execution.avg_trade_size",
                      "trade_metrics.execution.median_trade_size"],
            confidence=0.85))
    if leaves:
        out["capital_scale"] = _l2("Capital scale constraints", leaves)

    # --- 流动性 -------------------------------------------------------------
    leaves = []
    cash = cap.get("cash_buffer_ratio")
    if cash is not None:
        leaves.append(leaf(
            "liquidity_buffer", "Cash buffer constraint",
            f"Average cash share {_pct(cash)}"
            + (", the buffer is sufficient" if cash >= 0.3 else
               ", the buffer is relatively thin and sensitive to unexpected expenses" if cash < 0.1 else ""),
            evidence=[f"position_dynamics.capital_management.cash_buffer_ratio={cash}"],
            confidence=0.8))
    flow = asset.get("external_flow_net")
    if asset.get("external_flow_events"):
        leaves.append(leaf(
            "external_flow", "External fund inflows and outflows",
            f"Detected {asset['external_flow_events']} external deposits/withdrawals, net {_n(flow, 0)} (not investment returns)",
            evidence=["asset.external_flow_events", "asset.external_flow_net"],
            confidence=0.8,
            caveat="Deposits and withdrawals contaminate returns; time-weighted equity has been used to exclude them."))
    elif asset.get("available"):
        # Only when the asset block EXISTS can "none detected" be a finding.
        # Emitting it on a missing block would turn "we could not look" into
        # "there was nothing to find" — the zero-fill trap.
        leaves.append(leaf(
            "external_flow", "External fund inflows and outflows",
            "No external deposits or withdrawals detected during the observation window; account funding structure is stable",
            evidence=["asset.external_flow_events=0"], confidence=0.75))
    if leaves:
        # Distinct from the KYC `liquidity` key. A COLLIDING key is dropped by the
        # merge (`if l2_key in l2_out: continue`), so two different facts under one
        # name means one of them silently disappears. Names must be unique per L1.
        out["liquidity_constraints"] = _l2("Liquidity constraints", leaves)

    # --- 仓位约束 -----------------------------------------------------------
    leaves = []
    top1 = cap.get("max_invested_ratio")
    mw = conv.get("max_position_weight_ever")
    if mw is not None:
        leaves.append(leaf(
            "single_name_ceiling", "Per-instrument cap (actually reached)",
            f"Historical maximum position in a single instrument {_pct(mw)}"
            + (", close to a full position in a single instrument, with no diversification constraint in practice" if mw >= 0.9 else
               ", single-instrument position constrained to around 50%" if mw >= 0.45 else ", relatively restrained single-instrument cap"),
            evidence=[f"position_dynamics.conviction.max_position_weight_ever={mw}"],
            confidence=0.8))
    np_max = (ev.get("position") or {}).get("max_num_positions")
    if np_max is not None:
        leaves.append(leaf(
            "breadth_constraint", "Position count range",
            f"Held up to {np_max} instruments simultaneously",
            evidence=[f"position.max_num_positions={np_max}"], confidence=0.85))
    if cap.get("risk_on_off_switch_count") is not None:
        leaves.append(leaf(
            "exposure_swing", "Position adjustment magnitude",
            f"Invested ratio range {_pct(cap.get('min_invested_ratio'))}–{_pct(cap.get('max_invested_ratio'))}, risk on/off switches {cap['risk_on_off_switch_count']} times",
            evidence=["position_dynamics.capital_management.invested_ratio_range",
                      "position_dynamics.capital_management.risk_on_off_switch_count"],
            confidence=0.8))
    if leaves:
        out["position_limits"] = _l2("Position constraints", leaves)

    # --- 可用工具 -----------------------------------------------------------
    leaves = []
    lev = (ev.get("trade") or {}).get("leveraged_etf_trade_share")
    if lev is not None:
        leaves.append(leaf(
            "instrument_access", "Scope of tools actually used",
            f"Leverage/inverse products' share of trading volume {_pct(lev)}"
            + (", available instruments include leveraged derivatives" if lev > 0.02 else ", mainly holds spot"),
            evidence=[f"trade.leveraged_etf_trade_share={lev}"], confidence=0.8))
    if act.get("trading_days") is not None:
        leaves.append(leaf(
            "trading_window", "Tradable hours",
            f"{act['trading_days']} trading days had transactions during the observation window",
            evidence=[f"trade_metrics.activity.trading_days={act['trading_days']}"],
            confidence=0.8))
    if leaves:
        out["instrument_access"] = _l2("Tools and channel constraints", leaves)

    # --- 注意力带宽 ---------------------------------------------------------
    leaves = []
    b = ev.get("behavior") or {}
    bi = ev.get("behavior_intent") or {}
    tot = bi.get("totals") or {}
    if b.get("active_days") is not None and (b.get("date_range") or [None, None])[0]:
        leaves.append(leaf(
            "attention_bandwidth", "Attention bandwidth",
            f"{b['active_days']} active days / {n_days(b) }-day observation window"
            + (f", {_n(tot.get('avg_events_per_active_day'))} behavioral events per day"
               if tot.get("avg_events_per_active_day") is not None else ""),
            evidence=["behavior.active_days", "behavior.event_count",
                      "behavior_intent.totals.active_days"],
            confidence=0.8))
    rd = bi.get("research_depth") or {}
    if rd.get("median_session_minutes") is not None:
        leaves.append(leaf(
            "session_bandwidth", "Duration of single investment",
            f"Median {_n(rd['median_session_minutes'])} minutes per research session"
            + (f", longest {_n(rd.get('max_session_minutes'))} minutes"
               if rd.get("max_session_minutes") is not None else ""),
            evidence=["behavior_intent.research_depth.median_session_minutes"],
            confidence=0.75))
    if leaves:
        out["attention_bandwidth"] = _l2("Time and attention constraints", leaves)

    # --- 可见性 -------------------------------------------------------------
    # The visibility statement is a property of the PLATFORM, not of this user, so
    # it is true whenever we have any account data at all — and meaningless (and
    # misleading, as "we looked and found nothing") when we have none.
    out["data_visibility"] = {}
    if mv is not None or asset.get("available") or cap.get("used_days"):
        leaves = [leaf(
            "data_visibility", "Data visibility scope",
            "The profile covers only accounts visible on this platform; external assets, liabilities, and holdings at other institutions are not visible",
            evidence=["Only covers accounts visible on this platform; external assets are not visible"],
            confidence=1.0,
            caveat="Therefore, \"total assets\" and \"position ratio\" are both based on this platform's definition, not the user's total wealth.")]
        dq = pd.get("data_quality") or {}
        if dq.get("usable_ratio") is not None:
            leaves.append(leaf(
                "data_completeness", "Completeness of visible data",
                f"Position data availability {_pct(dq['usable_ratio'])}"
                + (f"({dq.get('carried_forward_days')} days missing, previous day's position carried forward)"
                   if dq.get("carried_forward_days") else ""),
                evidence=["position_dynamics.data_quality.usable_ratio"],
                confidence=0.9,
                caveat=("Below 50%, all conclusions based on positions are unreliable"
                        if (dq.get("usable_ratio") or 1) < 0.5 else None)))
        out["data_visibility"] = _l2("Visibility and data constraints", leaves)
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# 风险偏好 — four separate questions, never merged into one label
# ---------------------------------------------------------------------------
def _risk_preference(ev):
    pd = ev.get("position_dynamics") or {}
    asset = ev.get("asset") or {}
    pos = ev.get("position") or {}
    tm = ev.get("trade_metrics") or {}
    ac = ev.get("attention_capital") or {}
    conv = ev.get("conversation") or {}
    dims = ev.get("dimensions") or {}
    cap = pd.get("capital_management") or {}
    style = pd.get("style") or {}
    convic = pd.get("conviction") or {}
    derisk = pd.get("de_risking_speed") or {}
    ex = tm.get("execution") or {}

    out = {}

    # --- 观测承受能力 -------------------------------------------------------
    leaves = []
    dd = asset.get("max_drawdown")
    if dd is not None and asset.get("return_reliable"):
        sev = ("Extremely deep" if dd <= -0.35 else "Deeper" if dd <= -0.2 else
               "Moderate" if dd <= -0.1 else "Shallower")
        leaves.append(leaf(
            "realised_drawdown", "Deep drawdown actually experienced",
            f"Maximum drawdown {_pct(dd)} ({sev})"
            + (f", occurred on {asset.get('max_drawdown_date')}"
               if asset.get("max_drawdown_date") else "")
            + (f", recovered by {asset.get('recovered_by')}"
               if asset.get("recovered_by") else ", has not yet recovered"),
            evidence=["asset.max_drawdown", "asset.max_drawdown_date",
                      "asset.recovered_by"],
            confidence=0.85))
    vol = asset.get("equity_volatility")
    if vol is not None:
        leaves.append(leaf(
            "realised_volatility", "Volatility actually endured",
            f"Daily net asset value volatility {_pct(vol)}"
            + (", volatility is very high" if vol > 0.025 else ", moderate volatility" if vol > 0.012 else ", volatility is relatively low"),
            evidence=["asset.equity_volatility"], confidence=0.8))
    if pos.get("days_with_negative_unrealized_profit") is not None:
        leaves.append(leaf(
            "stress_exposure_days", "Unrealized-loss exposure pressure",
            f"{pos['days_with_negative_unrealized_profit']} days in an unrealized-loss state"
            + (f"(Accounts for {_pct(_ratio(pos['days_with_negative_unrealized_profit'], pos.get('days')))} of the observation window)"
               if pos.get("days") else ""),
            evidence=["position.days_with_negative_unrealized_profit", "position.days"],
            confidence=0.8))
    if leaves:
        out["observed_capacity"] = _l2(
            "Observed tolerance capacity", leaves,
            "What the user has actually endured—not questionnaire-based tolerance.")

    # --- 结构性敞口 ---------------------------------------------------------
    leaves = []
    occ = ev.get("ocf") or {}
    lev_mv = occ.get("leveraged_share")
    if lev_mv is not None:
        leaves.append(leaf(
            "leverage_exposure", "Leverage/derivatives exposure",
            f"Leverage/inverse products as a share of securities market value {_pct(lev_mv)}"
            + (", structurally amplifies risk" if lev_mv >= 0.2 else
               ", used sparingly" if lev_mv > 0 else ", unused"),
            evidence=["kyc_features.strategy_bucket_structure.value.leveraged_share"],
            confidence=0.8))
    top1 = (ev.get("kyc_alloc") or {}).get("top1")
    mw = convic.get("max_position_weight_ever")
    if mw is not None:
        leaves.append(leaf(
            "concentration_exposure", "Concentration exposure",
            f"Maximum single-instrument position {_pct(mw)}, average holdings {_n(convic.get('distinct_tickers_held'), 0)} instruments",
            evidence=["position_dynamics.conviction.max_position_weight_ever",
                      "position_dynamics.conviction.distinct_tickers_held"],
            confidence=0.8))
    if occ.get("directional_share") is not None:
        leaves.append(leaf(
            "directional_exposure", "Directional exposure",
            f"Directional stocks/themes account for {_pct(occ['directional_share'])}, broad-based/fixed-income core holdings account for {_pct(occ.get('core_share'))}",
            evidence=["kyc_features.strategy_bucket_structure"],
            confidence=0.75))
    if cap.get("risk_on_off_switch_count") is not None:
        leaves.append(leaf(
            "exposure_instability", "Exposure stability",
            f"Risk on/off switched {cap['risk_on_off_switch_count']} times, position adjustment range {_pct(cap.get('invested_ratio_range'))}"
            + (", exposure changes frequently" if cap["risk_on_off_switch_count"] >= 4
               else ", exposure is broadly stable"),
            evidence=["position_dynamics.capital_management.risk_on_off_switch_count"],
            confidence=0.8))
    if leaves:
        out["structural_exposure"] = _l2("Structural Exposure", leaves,
                                         "Where risk is actually placed, rather than how the user describes it.")

    # --- 意图与现实的偏离 ---------------------------------------------------
    leaves = []
    c_rate = (ac.get("conversion") or {}).get("research_to_holding_conversion")
    ori = ac.get("orientation") or {}
    if ori.get("nonheld_ticker_view_ratio") is not None:
        leaves.append(leaf(
            "attention_gap", "Attention-capital mismatch",
            f"{_pct(ori['nonheld_ticker_view_ratio'])} of behavior involved {ori.get('nonheld_tickers_attended')} instruments never held"
            + (f", and only {_pct(c_rate)} of these instruments were ultimately purchased" if c_rate is not None else ""),
            evidence=["attention_capital.orientation.nonheld_ticker_view_ratio",
                      "attention_capital.conversion.research_to_holding_conversion"],
            confidence=0.75))
    al = ac.get("attention_capital_alignment") or {}
    if al.get("attention_gt_capital"):
        tks = "、".join(al["attention_gt_capital"][:5])
        leaves.append(leaf(
            "watch_without_positions", "Instruments viewed but not purchased",
            f"{len(al['attention_gt_capital'])} instruments with attention exceeding capital allocation: {tks}",
            evidence=["attention_capital.attention_capital_alignment"
                      ".attention_gt_capital"], confidence=0.8))
    if al.get("capital_gt_attention"):
        tks = "、".join(al["capital_gt_attention"][:5])
        leaves.append(leaf(
            "positions_without_attention", "Instruments held but not viewed",
            f"{len(al['capital_gt_attention'])} instruments with capital allocation above attention level: {tks} (mostly passive or legacy positions)",
            evidence=["attention_capital.attention_capital_alignment"
                      ".capital_gt_attention"], confidence=0.8))
    if leaves:
        out["intent_reality_divergence"] = _l2(
            "Deviation between intent and reality", leaves,
            "Gap between attention, research, and actual capital actions; the closest evidence of \"true preferences\" that behavior can provide.")

    # --- 自述容忍度 ---------------------------------------------------------
    # The `users.base_profile_summary` channel was REMOVED because it carried
    # person-identifying text (see app/pii.py). That leaves no direct self-report of
    # risk tolerance anywhere in the data. Saying so is the honest answer; inventing
    # a number from behaviour would just be `observed_capacity` wearing a different
    # label, which is exactly the conflation this L2 exists to prevent.
    leaves = []
    if conv.get("available") and conv.get("count"):
        leaves.append(leaf(
            "stated_tolerance_proxy", "Stated risk intent (conversation topic proxy)",
            f"conversations has {conv['count']} conversations, including {conv.get('self_reference_count', 0)} self-referential/account-related conversations"
            + ("; The conversation title has no timestamp, so it cannot be determined whether it relates to a specific decision"
               if True else ""),
            status=PROXY,
            proxy_basis="Use topic tendencies in conversation titles as a proxy for stated risk appetite; titles have no timestamps or body text",
            confidence=min(PROXY_CONF_CAP, 0.4),
            evidence=["conversation.count", "conversation.self_reference_count"],
            caveat="This cannot determine risk tolerance; it only indicates whether the user has proactively discussed risk."))
    else:
        leaves.append(leaf(
            "stated_tolerance_unavailable", "Stated tolerance",
            "No self-report channel available: fields generating self-reports were removed in full because they contained personal identity information, and questionnaire data was not integrated. Therefore, \"how much risk the user says they can tolerate\" is unavailable from the current data.",
            status=GAP,
            evidence=["users.base_profile_summary was removed in v2.4 (contains personal identity information)"],
            caveat="Do not use behavioral tolerance capacity as a substitute for self-reported willingness—they are different facts."))
    out["stated_tolerance"] = _l2("Stated tolerance", leaves,
                                  "The user's self-declared risk willingness; with current data, conversation topics can only serve as a weak proxy.")

    # --- 工具偏好 -----------------------------------------------------------
    leaves = []
    b = ev.get("behavior") or {}
    if b.get("option_views") is not None:
        leaves.append(leaf(
            "instrument_appetite_options", "Attention to options-related instruments",
            f"Option-related page views {b['option_views']}"
            + (", actively focuses on derivatives" if b["option_views"] > 5 else ", pays almost no attention"),
            evidence=["behavior.option_views"], confidence=0.8))
    lev_trade = (ev.get("trade") or {}).get("leveraged_etf_trade_share")
    if lev_trade is not None:
        leaves.append(leaf(
            "instrument_appetite_leverage", "Actual use of leveraged products",
            f"Leverage/inverse products as a share of trades {_pct(lev_trade)}",
            evidence=["trade.leveraged_etf_trade_share"], confidence=0.8))
    etf = b.get("etf_views")
    if etf is not None and b.get("stock_views") is not None:
        tot = (etf or 0) + (b.get("stock_views") or 0)
        leaves.append(leaf(
            "instrument_appetite_breadth", "Individual stocks vs tool-type products",
            f"{b['stock_views']} stock views, {etf} ETF views"
            + (f"(ETF accounts for {_pct(_ratio(etf, tot))})" if tot else ""),
            evidence=["behavior.stock_views", "behavior.etf_views"],
            confidence=0.75))
    if leaves:
        out["observed_instrument_appetite"] = _l2(
            "Observed instrument preference", leaves,
            "The instruments the user is actually willing to engage with and the risk level are two separate dimensions.")

    # --- 亏损应对 -----------------------------------------------------------
    leaves = []
    adr = style.get("average_down_episode_rate")
    cl = style.get("close_loser_ratio")
    if adr is not None and style.get("accumulation_episodes") is not None:
        leaves.append(leaf(
            "loss_response_style", "Response to losses",
            f"Of the {style.get('accumulation_episodes')} add-on episodes, {style.get('average_down_episodes')} are averaging-down positions ({_pct(adr)})"
            + (", tends to average down" if adr >= 0.4 else ", moderate averaging-down tendency" if adr >= 0.15
               else ", rarely adds to positions while at an unrealized loss"),
            evidence=["position_dynamics.style.average_down_episode_rate",
                      "position_dynamics.outcome_response.depth_note"],
            confidence=0.8))
    depth = style.get("median_underwater_depth")
    if depth is not None:
        leaves.append(leaf(
            "loss_response_depth", "Averaging-down depth",
            f"At add-position, median underwater loss {_pct(depth)}, maximum {_pct(style.get('max_underwater_depth'))}"
            + (f";shallow underwater add-on positions account for {_pct(style.get('shallow_underwater_rate'))}"
               if style.get("shallow_underwater_rate") is not None else ""),
            evidence=["position_dynamics.style.median_underwater_depth",
                      "position_dynamics.style.max_underwater_depth"],
            confidence=0.8))
    if cl is not None:
        leaves.append(leaf(
            "loss_exit_tendency", "Disposition of losing trades",
            f"Loss-making positions among exits {_pct(cl)}"
            + (", tends to cut losses" if cl >= 0.6 else ", most exits occur when profitable"),
            evidence=["position_dynamics.style.close_loser_ratio"], confidence=0.75))
    if derisk.get("de_risking_episodes"):
        leaves.append(leaf(
            "de_risking_behaviour", "Proactive risk reduction",
            f"{derisk['de_risking_episodes']} de-risking episodes, median {_n(derisk.get('median_de_risking_days'))} days and magnitude {_pct(derisk.get('median_de_risking_magnitude'))}",
            evidence=["position_dynamics.de_risking_speed"], confidence=0.75))
    if leaves:
        out["loss_response"] = _l2("Response to losses and drawdowns", leaves,
                                   "What was actually done during drawdowns—more predictive of the next behavior than any risk label.")

    # --- 汇总等级（放在最后，作为结论而不是替代） ---------------------------
    rk = _dim(dims, "risk")
    if rk.get("revealed_risk_behavior"):
        leaves = [leaf(
            "revealed_level", "Disclosed risk level",
            f"{rk['revealed_risk_behavior']}"
            + (f"；{rk.get('narrative', '')[:80]}" if rk.get("narrative") else ""),
            evidence=["risk.revealed_risk_behavior"],
            confidence=rk.get("confidence"),
            caveat="This is a summary label for the above facts, not independent evidence; detailed dimensions take priority.")]
        out["revealed_level"] = _l2("Disclosed risk level (aggregate)", leaves)

    return out


# ---------------------------------------------------------------------------
# 投资逻辑 — how decisions are actually made
# ---------------------------------------------------------------------------
def _investment_logic(ev):
    bi = ev.get("behavior_intent") or {}
    tm = ev.get("trade_metrics") or {}
    pd = ev.get("position_dynamics") or {}
    ac = ev.get("attention_capital") or {}
    dims = ev.get("dimensions") or {}
    style = pd.get("style") or {}
    ex = tm.get("execution") or {}
    bt = bi.get("behavior_to_trade") or {}
    convic = pd.get("conviction") or {}
    forecast = ev.get("action_forecast") or {}

    out = {}

    # --- 主要策略 -----------------------------------------------------------
    leaves = []
    lg = _dim(dims, "logic")
    if lg.get("primary_style"):
        leaves.append(leaf("primary_style", "Primary style", str(lg["primary_style"]),
                           evidence=["investment_logic.primary_style"],
                           confidence=lg.get("confidence")))
    if lg.get("likely_styles"):
        leaves.append(leaf("likely_styles", "Candidate style",
                           "、".join(str(x) for x in lg["likely_styles"][:4]),
                           evidence=["investment_logic.likely_styles"],
                           confidence=lg.get("confidence")))
    if lg.get("thematic_focus"):
        leaves.append(leaf("thematic_focus", "Attention themes",
                           "、".join(str(x) for x in lg["thematic_focus"][:6]),
                           evidence=["investment_logic.thematic_focus"],
                           confidence=lg.get("confidence")))
    if leaves:
        out["primary_strategy"] = _l2("Primary strategy", leaves)

    # --- 信息依据 -----------------------------------------------------------
    leaves = []
    rs = bi.get("research_style") or {}
    order = [("news_driven", "新闻"), ("fundamental_driven", "Fundamentals"),
             ("technical_driven", "Technical analysis"), ("options_driven", "期权"),
             ("social_driven", "Social"), ("price_check_share", "Quote lookup")]
    ranked = [(zh, rs.get(k)) for k, zh in order if rs.get(k)]
    if ranked:
        ranked.sort(key=lambda x: -x[1])
        leaves.append(leaf(
            "signal_source_mix", "Information type composition",
            "、".join(f"{zh} {_pct(v)}" for zh, v in ranked[:4]),
            evidence=["behavior_intent.research_style"],
            confidence=0.8,
            caveat="This is \"what was viewed,\" not \"how deeply it was analyzed,\" and cannot be treated as research quality."))
    psh = rs.get("price_check_share")
    if psh is not None:
        leaves.append(leaf(
            "shallow_vs_deep", "Quote lookup-to-deep research ratio",
            f"Price checks as a share of views {_pct(psh)}"
            + (", browsing mostly stops at viewing prices" if psh >= 0.7 else
               ", browsing includes relatively more non-price content"),
            evidence=["behavior_intent.research_style.price_check_share"],
            confidence=0.75))
    rd = bi.get("research_depth") or {}
    if rd.get("research_grade_session_share") is not None:
        leaves.append(leaf(
            "research_grade", "Share of research-grade sessions",
            f"{_pct(rd['research_grade_session_share'])} of sessions reached research grade ({rd.get('sessions')} sessions in total)"
            + (", research is organized" if rd["research_grade_session_share"] >= 0.3
               else ", mostly just browsing"),
            evidence=["behavior_intent.research_depth.research_grade_session_share"],
            confidence=0.8))
    if leaves:
        out["signal_sources"] = _l2("Information basis", leaves,
                                    "What information is the decision based on—news, fundamentals, or price.")

    # --- 信念建立 -----------------------------------------------------------
    leaves = []
    entry = style.get("entry_style")
    if entry:
        zh = {"single_shot_entry": "One-time start of a new position", "staged_entry": "Staged position building"}.get(entry, entry)
        leaves.append(leaf("entry_style", "Position-opening method", zh,
                           evidence=["position_dynamics.style.entry_style"],
                           confidence=0.8))
    lat = (bt.get("research_to_trade_latency_hours") or {}).get("median")
    if lat is not None:
        leaves.append(leaf(
            "decision_speed", "Decision speed",
            f"Median lag from research to execution {_n(lat)} hours"
            + (", near-instant decision-making" if lat < 1 else
               ", decisions made within hours" if lat < 24 else ", spanning several days and constituting prudent decision-making"),
            evidence=["behavior_intent.behavior_to_trade"
                      ".research_to_trade_latency_hours.median"],
            confidence=0.75))
    pr = bt.get("prior_research_rate")
    if pr is not None:
        leaves.append(leaf(
            "research_before_trade", "Whether research occurred before the trade",
            f"{_pct(pr)} of trades had prior research within 48 hours"
            + (", generally researches before placing orders" if pr >= 0.6 else
               ", a considerable proportion is intraday improvisation" if pr < 0.3 else ""),
            evidence=["behavior_intent.behavior_to_trade.prior_research_rate"],
            confidence=0.8,
            caveat="Can only establish sequence of events, not that research caused the trade."))
    if leaves:
        out["conviction_building"] = _l2("Belief formation method", leaves)

    # --- 仓位逻辑 -----------------------------------------------------------
    leaves = []
    if convic.get("large_position_tolerance"):
        leaves.append(leaf(
            "position_sizing", "Single-instrument position tolerance",
            f"{convic['large_position_tolerance']}"
            + (f"(Historical maximum {_pct(convic.get('max_position_weight_ever'))})"
               if convic.get("max_position_weight_ever") is not None else ""),
            evidence=["position_dynamics.conviction.large_position_tolerance"],
            confidence=0.75))
    if style.get("pyramid_winner_rate") is not None:
        leaves.append(leaf(
            "add_direction_logic", "Add-direction preference",
            f"Of add-positions, {_pct(style['pyramid_winner_rate'])} were added to profitable positions and {_pct(style.get('average_down_rate'))} to underwater positions"
            + (", tends to add to winners" if (style["pyramid_winner_rate"] or 0) >
               (style.get("average_down_rate") or 0) else ", tends to average down losses"),
            evidence=["position_dynamics.style.pyramid_winner_rate",
                      "position_dynamics.style.average_down_rate"],
            confidence=0.8))
    if leaves:
        out["position_sizing_logic"] = _l2("Position sizing logic", leaves)

    # --- 退出逻辑 -----------------------------------------------------------
    leaves = []
    if style.get("full_exit_ratio") is not None:
        leaves.append(leaf(
            "exit_style", "Exit method",
            f"Full-position exits account for {_pct(style['full_exit_ratio'])} of exits, partial position trims {_pct(style.get('partial_exit_ratio'))}"
            + (", tends to close out in one go" if style["full_exit_ratio"] >= 0.6 else ", tends to gradually trim positions"),
            evidence=["position_dynamics.style.full_exit_ratio",
                      "position_dynamics.style.partial_exit_ratio"],
            confidence=0.8))
    if style.get("close_winner_ratio") is not None:
        leaves.append(leaf(
            "exit_trigger", "Exit trigger conditions",
            f"Profitable positions among exits {_pct(style['close_winner_ratio'])}, loss-making positions {_pct(style.get('close_loser_ratio'))}"
            + (", tends to take profits while retaining losses" if (style["close_winner_ratio"] or 0) >= 0.6
               else ", tends to stop out"),
            evidence=["position_dynamics.style.close_winner_ratio"],
            confidence=0.75))
    rr = (pd.get("conviction") or {}).get("reentry_rate")
    if rr is not None:
        leaves.append(leaf(
            "reentry_logic", "Re-entry tendency",
            f"Of instruments previously fully exited, {_pct(rr)} were bought again"
            + (", buys back previously held instruments" if rr >= 0.4 else ", rarely looks back after selling"),
            evidence=["position_dynamics.conviction.reentry_rate"], confidence=0.75))
    if leaves:
        out["exit_logic"] = _l2("Exit logic", leaves)

    # --- 逻辑持久度 ---------------------------------------------------------
    leaves = []
    tp = ac.get("theme_lead_lag") or {}
    for th in (tp.get("themes") or [])[:3]:
        if th.get("lag_days") is not None:
            leaves.append(leaf(
                f"theme_lag_{th.get('theme')}", f"Theme lag: {th.get('theme')}",
                f"Followed leading capital allocation by approximately {th['lag_days']} days (correlation {_n(th.get('correlation'), 3)}, based on {th.get('days_used')} days)",
                evidence=["attention_capital.theme_lead_lag"],
                confidence=0.6,
                caveat="Correlation does not equal causation and does not represent a trading decision path; the fewer sample days, the more likely the correlation is incidental."))
    pp = ac.get("persistence_pairing") or {}
    mix = pp.get("relationship_mix") or {}
    if mix:
        zh = {"core_conviction": "Long-term conviction holding", "watch_only": "Only considers non-held positions",
              "held_not_watched": "Held but infrequently viewed", "episodic_interest": "Intermittent attention",
              "active_trading_or_speculation": "Active trading/speculation"}
        leaves.append(leaf(
            "relationship_mix", "Composition of the relationship between attention and holdings",
            "、".join(f"{zh.get(k, k)} {v}" for k, v in
                     sorted(mix.items(), key=lambda kv: -kv[1])[:4]),
            evidence=["attention_capital.persistence_pairing.relationship_mix"],
            confidence=0.75))
    if leaves:
        out["thesis_durability"] = _l2("Logic persistence", leaves)

    # --- 下一步推演 ---------------------------------------------------------
    if forecast.get("available"):
        p = forecast.get("primary") or {}
        act = {"start_new_position": "Open a position in a new instrument", "add_to_existing": "Add to an existing position",
               "trim_position": "Trim", "exit_position": "Exit", "hold": "Remain unchanged"}
        leaves = [leaf(
            "next_action", "Next-action prediction",
            f"Based on the user's own habits, the next most likely action is [{act.get(p.get('action'), p.get('action'))}] (score {p.get('score')}, confidence {_pct(forecast.get('confidence'))})"
            + ("; possible instruments " + "、".join(t.get("ticker", "") for t in
                                        (p.get("targets") or [])[:5])
               if p.get("targets") else ""),
            evidence=["action_forecast.primary", "action_forecast.candidates",
                      "action_forecast.confidence"],
            confidence=forecast.get("confidence"),
            caveat=forecast.get("caveat"))]
        # the reasoning behind it, so the number is not taken on trust
        for s in (p.get("basis") or [])[:6]:
            if s.get("claim"):
                leaves.append(leaf(
                    f"next_action_basis_{s.get('name')}", f"Inference basis: {s.get('name')}",
                    str(s["claim"]),
                    evidence=[f"action_forecast..signal.{s.get('name')}"
                              f"={s.get('raw')}"],
                    confidence=None))
        if p.get("falsifier"):
            leaves.append(leaf(
                "next_action_falsifier", "Falsifiable conditions", str(p["falsifier"]),
                evidence=["action_forecast.primary.falsifier"], confidence=None,
                caveat="If the actual result matches this condition, this prediction is invalid."))
        out["next_action"] = _l2("Next-action reasoning", leaves,
                                 "Inference of future actions, not investment advice.")
    return out


# ---------------------------------------------------------------------------
# 投资者评估 — competence, discipline and bias, from behaviour
# ---------------------------------------------------------------------------
def _investor_assessment(ev):
    dims = ev.get("dimensions") or {}
    bi = ev.get("behavior_intent") or {}
    b = ev.get("behavior") or {}
    pd = ev.get("position_dynamics") or {}
    tm = ev.get("trade_metrics") or {}
    pos, ac = ev.get("position") or {}, ev.get("attention_capital") or {}
    conv = ev.get("conversation") or {}
    style = pd.get("style") or {}
    cap = pd.get("capital_management") or {}
    res = tm.get("results") or {}
    ex = tm.get("execution") or {}
    act = tm.get("activity") or {}
    asmt = _dim(dims, "assessment")

    out = {}

    # --- 经验层级 -----------------------------------------------------------
    leaves = []
    if asmt.get("experience_level"):
        leaves.append(leaf("experience_level", "Experience Level",
                           str(asmt["experience_level"]),
                           evidence=["investor_assessment.experience_level"],
                           confidence=asmt.get("confidence")))
    rd = bi.get("research_depth") or {}
    rb = bi.get("research_breadth") or {}
    if rd.get("sessions") is not None:
        leaves.append(leaf(
            "observable_experience", "Observable practical activity",
            f"{rd['sessions']} research sessions, covering {rb.get('unique_tickers')} instruments, {b.get('active_days')} active days",
            evidence=["behavior_intent.research_depth.sessions",
                      "behavior_intent.research_breadth.unique_tickers"],
            confidence=0.8))
    if act.get("total_orders") is not None:
        leaves.append(leaf(
            "execution_experience", "Actual trading experience",
            f"{act['total_orders']} trades / {act.get('trading_days')} trading days during the observation window, with {res.get('closed_trades')} trades closed",
            evidence=["trade_metrics.activity.total_orders",
                      "trade_metrics.results.closed_trades"],
            confidence=0.85))
    if leaves:
        # Distinct from the KYC `experience` key (see the collision note below).
        out["observable_experience"] = _l2("Experience Level", leaves)

    # --- 研究成熟度 ---------------------------------------------------------
    leaves = []
    if rb.get("attention_hhi") is not None:
        hhi = rb["attention_hhi"]
        leaves.append(leaf(
            "research_focus", "Research focus",
            f"Attention HHI {_n(hhi, 3)}"
            + (", highly concentrated in a few instruments" if hhi >= 0.5 else
               ", distributed across a relatively large number of instruments" if hhi < 0.15 else ", moderate concentration"),
            evidence=["behavior_intent.research_breadth.attention_hhi"],
            confidence=0.75))
    ee = bi.get("exploration_vs_exploitation") or {}
    if ee.get("new_ticker_event_share") is not None:
        held = ee.get("held_ticker_event_share")
        new = ee.get("new_ticker_event_share")
        leaves.append(leaf(
            "explore_vs_exploit", "Exploration and deepening",
            f"{_pct(new)} of behavior involved instruments not held, and {_pct(held)} involved instruments already held"
            + (", mainly seeks new opportunities" if (new or 0) > (held or 0) * 1.5 else
               ", mainly tracks existing holdings" if (held or 0) > (new or 0) * 1.5 else
               ", equal emphasis on both"),
            evidence=["behavior_intent.exploration_vs_exploitation"],
            confidence=0.75))
    ses = bi.get("sessions_top") or []
    if ses:
        top = ses[0]
        leaves.append(leaf(
            "deepest_session", "Most in-depth research",
            f'''{top.get('primary_ticker') or '—'}: {top.get('events')} events / {_n(top.get('minutes'))} minutes, reaching a maximum of "{top.get('max_intent_stage')}"''',
            evidence=["behavior_intent.sessions_top"], confidence=0.75))
    if leaves:
        out["research_sophistication"] = _l2("Research maturity", leaves)

    # --- 决策纪律 -----------------------------------------------------------
    leaves = []
    bt = bi.get("behavior_to_trade") or {}
    if bt.get("prior_research_rate") is not None:
        leaves.append(leaf(
            "plan_before_action", "Whether there was preparation before action",
            f"{_pct(bt['prior_research_rate'])} of trades had prior research within 48 hours",
            evidence=["behavior_intent.behavior_to_trade.prior_research_rate"],
            confidence=0.8))
    if cap.get("risk_on_off_switch_count") is not None:
        sw = cap["risk_on_off_switch_count"]
        leaves.append(leaf(
            "exposure_discipline", "Position discipline",
            f"Risk on/off switched {sw} times"
            + (", adjusts positions relatively conservatively" if sw <= 1 else
               ", frequent position changes, relatively weak discipline" if sw >= 4 else ", adjusts positions moderately"),
            evidence=["position_dynamics.capital_management"
                      ".risk_on_off_switch_count"],
            confidence=0.75))
    rot = pd.get("rotation") or {}
    if rot.get("composition_change_rate") is not None:
        leaves.append(leaf(
            "portfolio_drift", "Portfolio Drift",
            f"{_pct(rot['composition_change_rate'])} of trading days had position additions or removals (adjacent-day similarity {_pct(rot.get('avg_similarity_day_to_day'))})",
            evidence=["position_dynamics.rotation.composition_change_rate"],
            confidence=0.8))
    if leaves:
        out["decision_discipline"] = _l2("Decision discipline", leaves)

    # --- 行为偏差 -----------------------------------------------------------
    leaves = []
    em = _dim(dims, "emotion")
    if em.get("disposition_effect"):
        leaves.append(leaf("disposition_effect", "Disposition effect",
                           str(em["disposition_effect"]),
                           evidence=["emotion_and_psychology.disposition_effect"],
                           confidence=em.get("confidence")))
    if (style.get("shallow_underwater_rate") is not None
            and style.get("median_underwater_depth") is not None):
        leaves.append(leaf(
            "small_dip_adds", "Add to a position after a small floating loss",
            f"Among underwater additions, {_pct(style['shallow_underwater_rate'])} were less than 5% in magnitude (median depth {_pct(style['median_underwater_depth'])})"
            + (", mostly normal position additions rather than averaging down" if style["shallow_underwater_rate"] >= 0.5
               else ", mostly averaging down at genuine depth"),
            evidence=["position_dynamics.style.shallow_underwater_rate"],
            confidence=0.75))
    if em.get("chasing_momentum"):
        leaves.append(leaf("chasing_momentum", "Tendency to chase rallies",
                           str(em["chasing_momentum"]),
                           evidence=["emotion_and_psychology.chasing_momentum"],
                           confidence=em.get("confidence")))
    lca = ac.get("loss_conditioned_attention") or {}
    if lca.get("loss_attention_ratio") is not None:
        r = lca["loss_attention_ratio"]
        leaves.append(leaf(
            "loss_attention_ratio", "Attention tendency during unrealized losses",
            f"Attention ratio (loss/profit) {_n(r, 2)}"
            + (", views are significantly higher during unrealized losses" if r >= 1.5 else
               ", views are higher during unrealized gains" if r <= 0.67 else ", the two are close"),
            evidence=["attention_capital.loss_conditioned_attention"
                      ".loss_attention_ratio"],
            confidence=0.7,
            caveat=lca.get("interpretation_limit")))
    if leaves:
        out["behavioural_biases"] = _l2("Observable behavioral biases", leaves,
                                        "These are behavioral observations, not psychological diagnoses.")

    # --- 自我认知 -----------------------------------------------------------
    leaves = []
    if conv.get("self_reference_share") is not None:
        leaves.append(leaf(
            "self_awareness", "Degree of proactively understanding oneself",
            f"{_pct(conv['self_reference_share'])} of the conversation concerns the user's own accounts or holdings ({conv.get('self_reference_count')}/{conv.get('count')} items)",
            evidence=["conversation.self_reference_share"],
            confidence=0.7))
    if conv.get("asked_about_own_portfolio") is not None:
        n_asked = _cnt(conv["asked_about_own_portfolio"])
        leaves.append(leaf(
            "portfolio_questions", "Account-related questions",
            f"{n_asked} conversations directly asked about their own account status",
            evidence=["conversation.asked_about_own_portfolio"],
            confidence=0.6))
    if leaves:
        out["self_awareness"] = _l2("Self-awareness", leaves)

    # --- 学习轨迹 -----------------------------------------------------------
    leaves = []
    imp = pos.get("position_count_stability")
    if imp is not None:
        leaves.append(leaf(
            "behaviour_stability", "Behavior Stability",
            f"Position count stability {_pct(imp)}"
            + (", position structure is stable" if imp >= 0.7 else ", position structure changes substantially"),
            evidence=["position.position_count_stability"], confidence=0.75))
    if asmt.get("investment_status"):
        leaves.append(leaf("investment_status", "Investment status",
                           str(asmt["investment_status"]),
                           evidence=["investor_assessment.investment_status"],
                           confidence=asmt.get("confidence")))
    kn = asmt.get("knowledge_domains") or []
    if kn:
        leaves.append(leaf("knowledge_domains", "Knowledge domain",
                           "、".join(str(x) for x in kn[:6]),
                           evidence=["investor_assessment.knowledge_domains"],
                           confidence=asmt.get("confidence")))
    if leaves:
        out["learning_trajectory"] = _l2("Learning and status trajectory", leaves)

    return out


# ---------------------------------------------------------------------------
# 个性化与规则 — how this profile should be used
# ---------------------------------------------------------------------------
def _personalization_and_rules(ev):
    pd = ev.get("position_dynamics") or {}
    tm = ev.get("trade_metrics") or {}
    conv = ev.get("conversation") or {}
    ac = ev.get("attention_capital") or {}
    b = ev.get("behavior") or {}
    bi = ev.get("behavior_intent") or {}
    dims = ev.get("dimensions") or {}
    style = pd.get("style") or {}
    res = tm.get("results") or {}
    act = tm.get("activity") or {}

    out = {}

    # --- 可观测的决策规则 ---------------------------------------------------
    leaves = []
    adr = style.get("average_down_episode_rate")
    if adr is not None:
        leaves.append(leaf(
            "rule_loss_response", "Rule 1: Default action when at a loss",
            f"When unrealized losses are sufficiently large, {_pct(adr)} of add-position episodes become averaging-down episodes"
            + ("(Default: \"hold through\")" if adr >= 0.4 else "(Default: no action or watch)"),
            evidence=["position_dynamics.style.average_down_episode_rate"],
            confidence=0.7))
    if style.get("full_exit_ratio") is not None:
        leaves.append(leaf(
            "rule_exit", "Rule 2: Default action upon exit",
            f"When deciding to exit, {_pct(style['full_exit_ratio'])} choose to exit all at once"
            + ("(Default: \"exit completely\")" if style["full_exit_ratio"] >= 0.6
               else "(Default: \"trim gradually\")"),
            evidence=["position_dynamics.style.full_exit_ratio"], confidence=0.7))
    a2a = ac.get("attention_to_action") or {}
    if a2a.get("comparison_available"):
        r_with = a2a.get("add_rate_underwater_with_attention")
        r_without = a2a.get("add_rate_underwater_without_attention")
        leaves.append(leaf(
            "rule_attention_to_action", "Rule 3: Whether attention leads to adding",
            f"Add-to-position rate while at a loss: viewed {_pct(r_with)}, not viewed {_pct(r_without)}"
            + (", attention clearly precedes action" if (r_with or 0) > (r_without or 0) * 1.5
               else ", the difference is not significant"),
            evidence=["attention_capital.attention_to_action"],
            confidence=0.65,
            caveat=a2a.get("caveat")))
    if leaves:
        out["decision_rules"] = _l2(
            "Observable decision rules", leaves,
            "An if-then pattern generalized from repeated behaviors, not a rule stated by the user.")

    # --- 风控规则 -----------------------------------------------------------
    leaves = []
    cap = pd.get("capital_management") or {}
    if cap.get("exposure_style"):
        leaves.append(leaf("risk_rule_exposure", "Position control style",
                           str(cap["exposure_style"]),
                           evidence=["position_dynamics.capital_management"
                                     ".exposure_style"], confidence=0.75))
    derisk = pd.get("de_risking_speed") or {}
    if derisk.get("rapid_deployment_days") is not None:
        leaves.append(leaf(
            "risk_rule_deployment", "Position-opening speed",
            f"Number of days on which the investment ratio rose by ≥15% within {derisk['rapid_deployment_days']} days, indicating that adding to positions can be very fast",
            evidence=["position_dynamics.de_risking_speed.rapid_deployment_days"],
            confidence=0.7))
    ts = (ev.get("trading_digest") or {}).get("trade") or {}
    if ts.get("avg_down_hold_ratio") is not None:
        r = ts["avg_down_hold_ratio"]
        leaves.append(leaf(
            "risk_rule_hold_asymmetry", "Asymmetric holding periods for profitable and losing positions",
            f"Holding duration of losing trades ÷ winning trades = {_n(r, 2)}"
            + (", willing to hold through losses" if r > 1.5 else
               ", handles losses quickly" if r < 0.7 else ", the two are close"),
            evidence=["trading_digest.trade.avg_down_hold_ratio"],
            confidence=0.75))
    if leaves:
        out["risk_rules"] = _l2("Risk control and position rules", leaves)

    # --- 注意力分配规则 -----------------------------------------------------
    leaves = []
    ori = ac.get("orientation") or {}
    if ori.get("interpretation"):
        leaves.append(leaf(
            "attention_allocation", "Attention allocation pattern",
            f"{ori['interpretation']} (held instruments' share of attention {_pct(ori.get('held_ticker_view_ratio'))}, non-held instruments {_pct(ori.get('nonheld_ticker_view_ratio'))})",
            evidence=["attention_capital.orientation"], confidence=0.75))
    mix = (bi.get("event_mix") or [])[:4]
    if mix:
        leaves.append(leaf(
            "attention_actions", "Primary behavioral action",
            "、".join(f"{m.get('event_type')} {m.get('count')}" for m in mix
                     if m.get("event_type")),
            evidence=["behavior_intent.event_mix"], confidence=0.75))
    if b.get("short_timeframe_share") is not None:
        leaves.append(leaf(
            "attention_granularity", "Time scale for monitoring quotes",
            f"Short-term viewing share {_pct(b['short_timeframe_share'])}"
            + (", accustomed to watching short time horizons" if b["short_timeframe_share"] >= 0.5 else
               ", mainly focuses on longer time horizons"),
            evidence=["behavior.short_timeframe_share"], confidence=0.7))
    if leaves:
        out["attention_rules"] = _l2("Attention allocation rules", leaves)

    # --- 操作节奏 -----------------------------------------------------------
    leaves = []
    if act.get("trading_days") is not None and act.get("observation_days"):
        rate = _ratio(act["trading_days"], act["observation_days"])
        leaves.append(leaf(
            "trading_cadence", "Trading frequency",
            f"{act['observation_days']} days, with trades on {act['trading_days']} days ({_pct(rate)})"
            + (", trades almost every day" if (rate or 0) >= 0.6 else
               ", trades occasionally" if (rate or 0) <= 0.15 else ", intermittent trading"),
            evidence=["trade_metrics.activity.trading_days",
                      "trade_metrics.activity.observation_days"],
            confidence=0.8))
    if (ev.get("trade") or {}).get("median_gap_between_trading_days") is not None:
        leaves.append(leaf(
            "trading_gap", "Interval between two trades",
            f"Median interval between trading dates {_n((ev['trade'])['median_gap_between_trading_days'])} days",
            evidence=["trade.median_gap_between_trading_days"], confidence=0.75))
    if b.get("active_days") is not None:
        leaves.append(leaf(
            "browsing_cadence", "View frequency",
            f"{b['active_days']} active days, {b.get('event_count')} behavior events in total",
            evidence=["behavior.active_days", "behavior.event_count"],
            confidence=0.8))
    if leaves:
        out["operating_cadence"] = _l2("Operation and viewing rhythm", leaves)

    # --- 沟通偏好 -----------------------------------------------------------
    leaves = []
    if conv.get("chinese_share") is not None:
        leaves.append(leaf(
            "language_preference", "Language preference",
            f"Chinese accounts for {_pct(conv['chinese_share'])} of the conversation",
            evidence=["conversation.chinese_share"], confidence=0.7))
    th = conv.get("themes")
    if th:
        txt = _top(th, "theme", "count", limit=5)
        if txt:
            leaves.append(leaf(
                "topic_preference", "topics of interest", txt,
                evidence=["conversation.themes"], confidence=0.65,
                caveat="Conversation titles have no timestamps and can only indicate topics of interest, not association with a specific decision."))
    if leaves:
        # Distinct from the KYC `communication` key.
        out["communication_preferences"] = _l2("Communication and presentation preferences", leaves)

    return out


# ---------------------------------------------------------------------------
# 生命周期 / 背景 — what stage the user is at, observably
# ---------------------------------------------------------------------------
def _user_life_cycle(ev):
    b = ev.get("behavior") or {}
    tm = ev.get("trade_metrics") or {}
    pd = ev.get("position_dynamics") or {}
    ac = ev.get("attention_capital") or {}
    asset = ev.get("asset") or {}
    conv = ev.get("conversation") or {}
    act = tm.get("activity") or {}
    bi = ev.get("behavior_intent") or {}

    out = {}
    leaves = []
    # Observed stage, from activity rather than from a birth date we do not have.
    has_trade = bool((ev.get("trade") or {}).get("trade_count"))
    has_pos = bool((pd.get("capital_management") or {}).get("used_days"))
    has_view = bool(b.get("event_count"))
    if has_trade and has_pos:
        stage, why = "Active position-opening and trading stages", "Both transaction and position details are present"
    elif has_pos:
        stage, why = "Primarily holding, limited trading", "Held positions but no execution records"
    elif has_view:
        stage, why = "Observation/Learning Stage", "Browsing behavior only, with no positions or trades"
    else:
        stage = None
    if stage:
        leaves.append(leaf(
            "engagement_stage", "Current participation phase",
            f"{stage} (basis: {why})",
            evidence=["trade_metrics.activity.total_orders",
                      "position_dynamics.capital_management.used_days",
                      "behavior.event_count"], confidence=0.8))
    days = b.get("date_range") or [None, None]
    if days[0] and asset.get("num_days"):
        leaves.append(leaf(
            "account_maturity", "Account data span",
            f"Visible data spans {asset['num_days']} days ({days[0]} to {days[1]})",
            evidence=["asset.num_days", "behavior.date_range"], confidence=0.85))
    if leaves:
        out["engagement_stage"] = _l2("Participation phase", leaves,
                                      "Inferred from observable activity and holdings, not the life stage reported in a questionnaire.")

    # 活跃趋势：有数据才给
    leaves = []
    sessions = bi.get("sessions_top") or []
    if len(sessions) >= 2:
        first, last = sessions[-1], sessions[0]
        leaves.append(leaf(
            "activity_recency", "Recent activity intensity",
            f"Most recent concentrated session {str(last.get('start') or '')[:10]} ({last.get('events')} events); earliest {str(first.get('start') or '')[:10]}",
            evidence=["behavior_intent.sessions_top"], confidence=0.7))
    if conv.get("count") is not None:
        leaves.append(leaf(
            "conversation_volume", "Conversation activity",
            f"{conv['count']} conversations ({conv.get('unique_count')} after deduplication)",
            evidence=["conversation.count", "conversation.unique_count"],
            confidence=0.7))
    if leaves:
        out["activity_trend"] = _l2("Activity and trends", leaves)
    return out


def _background(ev):
    bi = ev.get("behavior_intent") or {}
    b = ev.get("behavior") or {}
    tm = ev.get("trade_metrics") or {}
    conv = ev.get("conversation") or {}
    act = tm.get("activity") or {}
    rd = bi.get("research_depth") or {}

    out = {}
    leaves = []
    if rd.get("sessions") is not None:
        leaves.append(leaf(
            "observable_experience", "Observable investment experience",
            f"{rd['sessions']} research sessions, {rd.get('total_minutes') and round(rd['total_minutes']) or 0} minutes of research, {act.get('total_orders') or 0} trades",
            evidence=["behavior_intent.research_depth.sessions",
                      "trade_metrics.activity.total_orders"], confidence=0.8))
    dom = bi.get("info_type_mix")
    if dom:
        txt = _top(dom, "info_type", "share")
        if txt:
            leaves.append(leaf(
                "knowledge_domains", "Knowledge domain (by browsing composition)", txt,
                evidence=["behavior_intent.info_type_mix"], confidence=0.7,
                caveat="Infer from the proportion of information types in browsing events, reflecting what was viewed, not the level of understanding."))
    if leaves:
        out["observable_experience"] = _l2("Observable experience and knowledge", leaves)

    leaves = []
    if b.get("top_viewed_tickers"):
        tks = [t.get("ticker") if isinstance(t, dict) else t
               for t in b["top_viewed_tickers"][:8]]
        leaves.append(leaf(
            "attention_universe", "Range of instruments under long-term attention",
            "、".join(str(t) for t in tks if t),
            evidence=["behavior.top_viewed_tickers"], confidence=0.75))
    themes = bi.get("themes")
    if themes:
        txt = _top(themes, "theme", "attention_share", limit=6)
        if txt:
            leaves.append(leaf(
                "theme_universe", "Thematic scope of attention", txt,
                evidence=["behavior_intent.themes"], confidence=0.7))
    if leaves:
        out["platform_usage"] = _l2("Platform usage and coverage", leaves)
    return out


# ---------------------------------------------------------------------------
# 情绪与心理 / App 行为 — extra depth on top of the existing L2
# ---------------------------------------------------------------------------
def _emotion_extra(ev):
    pd = ev.get("position_dynamics") or {}
    ac = ev.get("attention_capital") or {}
    tm = ev.get("trade_metrics") or {}
    style = pd.get("style") or {}
    out = {}
    leaves = []
    or_ = pd.get("outcome_response") or {}
    a2a = ac.get("attention_to_action") or {}
    if or_.get("average_down_tendency"):
        leaves.append(leaf(
            "outcome_response", "Reaction tendency after profit or loss occurs",
            f"Tendency to add after losses {or_['average_down_tendency']}, tendency to add after gains {or_.get('pyramid_winner_tendency')} (low-profit add-ons + high-loss averaging down is a typical disposition-effect combination)",
            evidence=["position_dynamics.outcome_response"], confidence=0.75))
    if a2a.get("interpretation"):
        leaves.append(leaf(
            "attention_to_action", "Whether attention leads to action",
            str(a2a["interpretation"]),
            evidence=["attention_capital.attention_to_action"],
            confidence=0.65, caveat=a2a.get("caveat")))
    if leaves:
        out["outcome_response_patterns"] = _l2("Profit and loss reaction pattern", leaves)
    return out


def _app_behavior_extra(ev):
    bi = ev.get("behavior_intent") or {}
    conv = ev.get("conversation") or {}
    out = {}
    leaves = []
    ses = bi.get("sessions_top") or []
    if ses:
        bits = []
        for r in ses[:3]:
            seg = f"{r.get('primary_ticker') or '—'}"
            if r.get("start"):
                seg += f"（{str(r['start'])[:10]}"
                if r.get("minutes") is not None:
                    seg += f", {_n(r['minutes'])} minutes"
                seg += "）"
            bits.append(seg)
        leaves.append(leaf(
            "session_patterns", "Typical research session",
            "；".join(bits),
            evidence=["behavior_intent.sessions_top"], confidence=0.75))
    if conv.get("themes"):
        txt = _top(conv["themes"], "theme", "count", limit=5)
        if txt:
            leaves.append(leaf(
                "conversation_themes", "Conversation topic distribution", txt,
                evidence=["conversation.themes"], confidence=0.65))
    if leaves:
        out["session_patterns"] = _l2("Session and topic patterns", leaves)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Which builder owns which L1. L1s not listed keep their existing KYC/dim sources.
BUILDERS = {
    "goals": _goals,
    "constraints": _constraints,
    "risk_preference": _risk_preference,
    "investment_logic": _investment_logic,
    "investor_assessment": _investor_assessment,
    "personalization_and_rules": _personalization_and_rules,
    "user_life_cycle": _user_life_cycle,
    "background": _background,
    "emotion_and_psychology": _emotion_extra,
    "app_behavior_analysis": _app_behavior_extra,
}


def build_l2(ev, dims=None):
    """`{l1_key: {l2_key: {label, hint, leaves}}}` from the deterministic evidence.

    `dims` are the LLM dimension outputs, passed so a summary leaf can cite them.
    Every builder is wrapped: a failure here must degrade that L1's depth, never the
    whole tree.
    """
    ev = dict(ev or {})
    if dims:
        ev["dimensions"] = dims
    out = {}
    for l1, fn in BUILDERS.items():
        try:
            blocks = fn(ev) or {}
        except Exception as e:                                        # noqa: BLE001
            # A crash must not masquerade as "no data for this user".
            blocks = {"_error": {"label": "Generation failed", "hint": None,
                                 "leaves": [leaf(
                                     "_error", "Segment Failure",
                                     f"Failed to generate dimension breakdown: {type(e).__name__}: {e}",
                                     status=GAP,
                                     evidence=["l2_taxonomy.build_l2"])]}}
        blocks = {k: v for k, v in blocks.items()
                  if v and any(x for x in (v.get("leaves") or [])
                               if x.get("status") in (FILLED, PROXY))}
        for v in blocks.values():
            v["leaves"] = [x for x in v.get("leaves") or []
                           if x.get("status") in (FILLED, PROXY)]
        if blocks:
            out[l1] = blocks
    from .english_output import englishize
    return englishize(out)


def count_leaves(l2):
    return sum(len(b.get("leaves") or []) for blocks in (l2 or {}).values()
               for b in blocks.values())


def window_label(ev, l1=None):
    """The observation window shared by these derivations, for headers."""
    pd = (ev or {}).get("position_dynamics") or {}
    d = pd.get("window_days")
    return f"Past {d} days" if d else None


def n_days(b):
    """Span of the behaviour window, so a rate is never reported without it."""
    dr = (b or {}).get("date_range") or [None, None]
    try:
        from .taxonomy import parse_ts
        a, z = parse_ts(dr[0]), parse_ts(dr[1])
        if a and z:
            return (z - a).days + 1
    except Exception:                                                 # noqa: BLE001
        pass
    return None
