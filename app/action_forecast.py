"""Deterministic forecast of what this user is likely to do next.

Why this exists
---------------
Everything else on the profile describes the past. A reader (and a downstream
advisor) wants the same evidence turned forward: given how this user has behaved,
what is the most likely next capital action, on which names, and how much should
that be trusted?

This module answers that from the SAME deterministic block the rest of the
profile uses — `attention_capital` (attention vs capital) and `position_dynamics`
(how positions are actually built and unwound). No LLM: a forecast that cannot be
recomputed is a horoscope.

How the score is built, and the one honesty rule that shapes it
---------------------------------------------------------------
Each candidate action is scored from a list of SIGNALS. Every signal has a weight,
a raw value read from a field, and a recorded contribution, so a score is literally
the sum of its stated parts and nothing enters it unannounced.

The rule that matters: **a missing signal lowers confidence, never the score.** The
natural implementation — treating an absent signal as 0 — punishes a user for our
lack of data: someone with no trades would score low on every action and be handed
a confident "hold", which is not a prediction, it is the absence of one. So the
score is the weighted mean over the signals that were AVAILABLE, and `coverage`
reports how much of the weight actually had data behind it. Below
`MIN_SIGNAL_COVERAGE` the candidate is reported unscored.

Every candidate also carries a `falsifier`: the observable that would show the
prediction was wrong. A forecast without one cannot be checked, and an unchecked
forecast is indistinguishable from a guess.

This is a description of the user's own habitual behaviour. It is NOT a market
view, NOT a recommendation, and NOT suitable for use as either.
"""
FORECAST_VERSION = "action-forecast-v2-en"

# Candidate actions, in the order they are reported when scores tie.
ACTIONS = ("start_new_position", "add_to_existing", "trim_position",
           "exit_position", "hold")

ACTION_ZH = {
    "start_new_position": "Start a new position",
    "add_to_existing": "Add to an existing position",
    "trim_position": "Trim a position",
    "exit_position": "Exit a position",
    "hold": "Hold",
}

# Weight of every signal, kept in one place so the model's shape is reviewable
# without reading the scoring code. Weights are relative within an action.
#
# `frequency` is deliberately the heaviest signal in EVERY action. The first version
# scored actions from STYLE ratios alone — `full_exit_ratio` (75% of exits are full
# closes) — and predicted 清仓 as the most likely next move for a user who had made
# 18 adds and 6 closes in the window. That conflates two different questions:
#
#   style      "when they exit, how do they exit?"        conditional, a ratio
#   propensity "how often do they do this at all?"        a rate per unit time
#
# Only the second answers "what will they do next", so it dominates, and the style
# ratios modulate it rather than substituting for it.
W = {
    "frequency": 1.5, "inactivity": 1.2,
    "pool_size": 1.0, "pool_depth": 0.8, "acted_on_research": 0.9,
    "research_before_trade": 0.7, "active_explorer": 0.5,
    "add_habit": 0.9, "underwater_add_habit": 1.0, "deep_underwater_habit": 0.6,
    "still_accumulating": 0.5, "trim_habit": 0.7, "de_risk_history": 0.8,
    "partial_exit_style": 0.6, "full_exit_habit": 0.9, "closes_from_loss": 0.6,
    "book_stability": 1.0, "deployment_slowness": 0.8, "position_continuity": 0.7,
}

# Reference rates for normalising an action count into [0,1], per trading day.
# Stated rather than tuned per user: `RATE_CEILING` is "about as often as an active
# retail book does this", and it is the same for everyone so two users' scores mean
# the same thing.
RATE_CEILING = {"open": 0.15, "add": 0.15, "reduce": 0.10, "close": 0.10}
# How many observed actions make a solid behavioural record for a CONFIDENT
# prediction. A user with 3 actions in the window may act again, but "this is their
# habit" is not supportable at that sample, however complete our signals are.
SAMPLE_FOR_CONFIDENCE = 20

# Total-activity ceiling for the `hold` prediction: at or above this the user is
# doing something often enough that "no change" is not the best guess.
INACTIVITY_CEILING = 0.30

# A score built on too little of its own weight is not a prediction. Below this the
# candidate is reported with `score: None` and a stated reason.
MIN_SIGNAL_COVERAGE = 0.5

# Where a pattern must be observed before it counts as a habit rather than an
# accident. Same discipline as MIN_EPISODES_FOR_STYLE elsewhere.
MIN_EVENTS_FOR_HABIT = 3

# The forecast is about the user's NEAR-TERM behaviour, so the horizon is stated
# explicitly and derived from the observation window rather than invented.
HORIZON_NOTE = ("The forecast horizon equals the observation-window length and defines "
                "what near term means. It describes habitual user action, not a market view.")


def _ratio(num, den):
    if num is None or not den:
        return None
    return num / den


def _clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _norm(v, lo, hi):
    """Map a value into [0,1] against a stated range, or None when unavailable."""
    if v is None:
        return None
    if hi <= lo:
        return None
    return _clip((v - lo) / (hi - lo))


def _sig(signals, name, raw, points):
    """Record one signal. `points` is None when the raw value was unavailable."""
    signals.append({"name": name, "weight": W.get(name, 1.0), "raw": raw,
                    "points": points})


def _score(signals):
    """Weighted mean over AVAILABLE signals, plus how much weight had data.

    See the module docstring: absent signals must not be scored as zero.
    """
    total = sum(s["weight"] for s in signals)
    used = sum(s["weight"] for s in signals if s["points"] is not None)
    if not used:
        return None, 0.0
    earned = sum(s["weight"] * s["points"] for s in signals
                 if s["points"] is not None)
    return round(earned / used, 3), round(used / total, 3)


def _targets_from_pool(pool, pair_by_ticker, limit=5):
    """Rank the watched-not-bought pool into named candidates."""
    out = []
    for t in (pool or []):
        meta = pair_by_ticker.get(t) or {}
        days = meta.get("attention_days")
        out.append({"ticker": t, "attention_days": days,
                    "why": (f"Viewed on {days} days during the window but never opened a position"
                            if days is not None
                            else "Viewed during the window but never opened a position")})
    out.sort(key=lambda x: (x["attention_days"] is None, -(x["attention_days"] or 0)))
    return out[:limit]


def build_forecast(attention_capital, position_dynamics):
    """Rank the user's likely next capital actions. Deterministic and auditable."""
    ac = attention_capital or {}
    pd = position_dynamics or {}
    ac_ok = bool(ac.get("available"))
    pd_ok = bool(pd.get("available"))
    if not ac_ok and not pd_ok:
        return {"available": False, "version": FORECAST_VERSION,
                "reason": "Attention-capital alignment and position dynamics are both unavailable; no behavioral next-action inference is possible"}

    align = ac.get("attention_capital_alignment") or {}
    conv = ac.get("conversion") or {}
    pre = ac.get("pre_entry") or {}
    ori = ac.get("orientation") or {}
    pair = ac.get("persistence_pairing") or {}
    style = pd.get("style") or {}
    cap = pd.get("capital_management") or {}
    derisk = pd.get("de_risking_speed") or {}
    rot = pd.get("rotation") or {}
    convic = pd.get("conviction") or {}
    outcome = pd.get("outcome_response") or {}
    states = align.get("state_counts") or {}

    pair_by_ticker = {t.get("ticker"): t for t in (pair.get("tickers") or [])
                      if t.get("ticker")}
    pool = list(align.get("attention_gt_capital") or [])
    held = {t.get("ticker") for t in (pair.get("tickers") or [])}

    adds = style.get("adds")
    opens = style.get("opens")
    reduces = style.get("reduces")
    closes = style.get("closes")
    win_days = pd.get("window_days") or ac.get("window_days")

    # Propensity per trading day. This is the backbone of every prediction; see the
    # comment on W above for why style ratios must not replace it.
    def _rate(n):
        if n is None or not win_days:
            return None
        return n / win_days

    rates = {"open": _rate(opens), "add": _rate(adds),
             "reduce": _rate(reduces), "close": _rate(closes)}
    total_rate = (None if all(v is None for v in rates.values())
                  else sum(v or 0.0 for v in rates.values()))

    def _counts_claim(kind):
        n = {"open": opens, "add": adds, "reduce": reduces,
             "close": closes}[kind]
        if n is None or not win_days:
            return None
        return f"{n} actions / {win_days} days = {n / win_days:.3f} actions/day"

    candidates = []

    # ---- 建仓新标的 ---------------------------------------------------------
    sig = []
    _sig(sig, "frequency", rates["open"],
         _norm(rates["open"], 0, RATE_CEILING["open"]))
    n_pool = len(pool) if pool else None
    _sig(sig, "pool_size", n_pool, _norm(n_pool, 0, 6))
    pool_days = [pair_by_ticker.get(t, {}).get("attention_days") for t in pool]
    pool_days = [d for d in pool_days if d is not None]
    med_pool_days = sorted(pool_days)[len(pool_days) // 2] if pool_days else None
    _sig(sig, "pool_depth", med_pool_days, _norm(med_pool_days, 0, 30))
    # Do they actually pull the trigger on what they study? This is the single most
    # relevant precedent: a user who studies 6 names and buys 0 of them is unlikely
    # to start now, however large the pool looks.
    c_rate = conv.get("research_to_holding_conversion")
    _sig(sig, "acted_on_research", c_rate, _norm(c_rate, 0, 0.5))
    _sig(sig, "research_before_trade", pre.get("prior_attention_rate_7d"),
         _norm(pre.get("prior_attention_rate_7d"), 0, 1))
    _sig(sig, "active_explorer", ori.get("nonheld_ticker_view_ratio"),
         _norm(ori.get("nonheld_ticker_view_ratio"), 0, 1))
    start_score, start_cov = _score(sig)
    start_targets = _targets_from_pool(pool, pair_by_ticker)
    if start_score is not None and start_cov >= MIN_SIGNAL_COVERAGE:
        candidates.append({
            "action": "start_new_position", "action_zh": ACTION_ZH["start_new_position"],
            "score": start_score, "signal_coverage": start_cov,
            "targets": start_targets,
            "basis": _basis(sig, {
                "frequency": _counts_claim("open"),
                "pool_size": f"{n_pool} instruments were viewed but never held during the window",
                "pool_depth": (f"Median attention in the candidate pool: {med_pool_days} days"
                               if med_pool_days is not None else None),
                "acted_on_research": (f"Share of researched instruments eventually purchased: "
                                      f"{c_rate:.1%}" if c_rate is not None else None),
                "research_before_trade": (f"Share of executions preceded by research within 48h: "
                                          f"{pre.get('prior_attention_rate_7d'):.1%}"
                                          if pre.get("prior_attention_rate_7d")
                                          is not None else None),
                "active_explorer": (f"Share of views directed to non-held instruments: "
                                    f"{ori.get('nonheld_ticker_view_ratio'):.1%}"
                                    if ori.get("nonheld_ticker_view_ratio")
                                    is not None else None),
            }),
            "sample_note": (f"Candidate pool contains {n_pool} instruments"
                            if n_pool is not None else "No candidate-pool data"),
            "falsifier": ("This prediction is falsified if no candidate-pool instrument becomes a new holding during the forecast window"),
        })

    # ---- 加仓 --------------------------------------------------------------
    sig = []
    _sig(sig, "frequency", rates["add"],
         _norm(rates["add"], 0, RATE_CEILING["add"]))
    add_prop = _ratio(adds, opens) if (adds is not None and opens) else None
    _sig(sig, "add_habit", add_prop, _norm(add_prop, 0, 3))
    adr = style.get("average_down_episode_rate")
    _sig(sig, "underwater_add_habit", adr, _norm(adr, 0, 0.5))
    _sig(sig, "deep_underwater_habit", style.get("median_underwater_depth"),
         _norm(style.get("median_underwater_depth"), 0, 0.2))
    # Is the book still being built, or is it being unwound? A user whose episodes
    # are mostly accumulative has a different next move from one who is distributing.
    acc, dist = style.get("accumulation_episodes"), style.get("distribution_episodes")
    building = None if acc is None else (1.0 if acc > (dist or 0) else 0.0)
    _sig(sig, "still_accumulating",
         (None if acc is None else {"accumulation": acc, "distribution": dist}),
         building)
    add_score, add_cov = _score(sig)
    add_tendency = outcome.get("average_down_tendency")
    if add_score is not None and add_cov >= MIN_SIGNAL_COVERAGE:
        candidates.append({
            "action": "add_to_existing", "action_zh": ACTION_ZH["add_to_existing"],
            "score": add_score, "signal_coverage": add_cov,
            # they add to what they already hold and monitor; the disagreement
            # names are the natural targets
            "targets": _held_targets(held, pair_by_ticker, align),
            "basis": _basis(sig, {
                "frequency": _counts_claim("add"),
                "add_habit": (f"{adds} adds versus {opens} new positions" if opens else None),
                "underwater_add_habit": (f"Average-down episodes as a share of add episodes: "
                                         f"{adr:.1%}" if adr is not None else None),
                "deep_underwater_habit": (f"Median unrealized loss at the time of an add: "
                                          f"{style.get('median_underwater_depth'):.1%}"
                                          if style.get("median_underwater_depth")
                                          is not None else None),
                "still_accumulating": (
                    f"{style.get('accumulation_episodes')} accumulation episodes versus "
                    f"{style.get('distribution_episodes')} distribution episodes"
                    if style.get("accumulation_episodes") is not None else None),
            }),
            "sample_note": (f"{adds} adds and {opens} new positions; "
                            f"loss-response tendency={add_tendency}"
                            if adds is not None else "No add records"),
            "falsifier": ("This prediction is falsified if no existing position is increased by at least 20% relative to the prior day during the forecast window"),
        })

    # ---- 减仓 / 清仓 --------------------------------------------------------
    sig = []
    _sig(sig, "frequency", rates["reduce"],
         _norm(rates["reduce"], 0, RATE_CEILING["reduce"]))
    trim_prop = _ratio(reduces, (reduces or 0) + (closes or 0)) \
        if (reduces is not None and closes is not None) else None
    _sig(sig, "trim_habit", trim_prop, _norm(trim_prop, 0, 0.5))
    _sig(sig, "de_risk_history", derisk.get("de_risking_episodes"),
         _norm(derisk.get("de_risking_episodes"), 0, 4))
    _sig(sig, "partial_exit_style", style.get("partial_exit_ratio"),
         _norm(style.get("partial_exit_ratio"), 0, 0.5))
    trim_score, trim_cov = _score(sig)
    if trim_score is not None and trim_cov >= MIN_SIGNAL_COVERAGE:
        candidates.append({
            "action": "trim_position", "action_zh": ACTION_ZH["trim_position"],
            "score": trim_score, "signal_coverage": trim_cov,
            "targets": _held_targets(held, pair_by_ticker, align, prefer="heavy"),
            "basis": _basis(sig, {
                "frequency": _counts_claim("reduce"),
                "trim_habit": (f"{reduces} trims versus {closes} exits"
                               if reduces is not None and closes is not None else None),
                "de_risk_history": (f"{derisk.get('de_risking_episodes')} de-risking episodes; "
                                    f"median magnitude {derisk.get('median_de_risking_magnitude')}"
                                    if derisk.get("de_risking_episodes") is not None else None),
                "partial_exit_style": (f"Partial trims account for {style.get('partial_exit_ratio'):.1%}"
                                       if style.get("partial_exit_ratio") is not None
                                       else None),
            }),
            "sample_note": (f"{reduces} trims and {closes} exits"
                            if reduces is not None else "No trim records"),
            "falsifier": ("This prediction is falsified if no existing position is trimmed during the forecast window"),
        })

    sig = []
    _sig(sig, "frequency", rates["close"],
         _norm(rates["close"], 0, RATE_CEILING["close"]))
    fer = style.get("full_exit_ratio")
    _sig(sig, "full_exit_habit", fer, _norm(fer, 0, 0.6))
    _sig(sig, "closes_from_loss", style.get("close_loser_ratio"),
         _norm(style.get("close_loser_ratio"), 0, 1))
    _sig(sig, "de_risk_history", derisk.get("de_risking_episodes"),
         _norm(derisk.get("de_risking_episodes"), 0, 4))
    exit_score, exit_cov = _score(sig)
    if exit_score is not None and exit_cov >= MIN_SIGNAL_COVERAGE:
        candidates.append({
            "action": "exit_position", "action_zh": ACTION_ZH["exit_position"],
            "score": exit_score, "signal_coverage": exit_cov,
            "targets": _held_targets(held, pair_by_ticker, align, prefer="light"),
            "basis": _basis(sig, {
                "frequency": _counts_claim("close"),
                "full_exit_habit": (f"Full exits account for {fer:.1%} of trims plus exits"
                                    if fer is not None else None),
                "closes_from_loss": (f"Share of exits closed at a loss: "
                                     f"{style.get('close_loser_ratio'):.1%}"
                                     if style.get("close_loser_ratio") is not None else None),
                "de_risk_history": (f"{derisk.get('de_risking_episodes')} de-risking episodes"
                                    if derisk.get("de_risking_episodes") is not None else None),
            }),
            "sample_note": (f"{closes} exits" if closes is not None else "No exit records"),
            "falsifier": "This prediction is falsified if no position is fully closed during the forecast window",
        })

    # ---- 维持不变 ----------------------------------------------------------
    # The complement of activity: a book that barely moves makes "no action" the
    # honest best guess, and one that moves daily does not.
    sig = []
    _sig(sig, "inactivity",
         (None if total_rate is None else round(total_rate, 4)),
         (None if total_rate is None
          else 1 - _clip(total_rate / INACTIVITY_CEILING)))
    sim = rot.get("avg_similarity_day_to_day")
    _sig(sig, "book_stability", sim, _norm(sim, 0.5, 1.0))
    switch = cap.get("risk_on_off_switch_count")
    _sig(sig, "deployment_slowness",
         (None if switch is None else max(0.0, 1 - switch / 6.0)),
         (None if switch is None else max(0.0, 1 - switch / 6.0)))
    pers = convic.get("avg_persistence_ratio")
    _sig(sig, "position_continuity", pers, _norm(pers, 0, 0.8))
    hold_score, hold_cov = _score(sig)
    if hold_score is not None and hold_cov >= MIN_SIGNAL_COVERAGE:
        candidates.append({
            "action": "hold", "action_zh": ACTION_ZH["hold"],
            "score": hold_score, "signal_coverage": hold_cov,
            "targets": _held_targets(held, pair_by_ticker, align, prefer="heavy"),
            "basis": _basis(sig, {
                "inactivity": (f"{sum(v for v in rates.values() if v)} actions/day in the window (opens + adds + trims + exits)" if total_rate is not None else None),
                "book_stability": (f"Adjacent-trading-day holding-set similarity: {sim:.1%}"
                                   if sim is not None else None),
                "deployment_slowness": (f"{switch} risk-on/risk-off switches"
                                        if switch is not None else None),
                "position_continuity": (f"Average instrument holding persistence: {pers:.1%}"
                                        if pers is not None else None),
            }),
            "sample_note": (f"Holding composition changed on {rot.get('composition_change_days')} of {win_days} days" if rot.get("composition_change_days")
                            is not None else "No holding-rotation data"),
            "falsifier": ("This prediction is falsified if the holding set changes during the forecast window through any new or fully closed position"),
        })

    if not candidates:
        return {"available": False, "version": FORECAST_VERSION,
                "reason": (f"Insufficient available signals: each action requires at least {MIN_SIGNAL_COVERAGE:.0%} weighted signal coverage; no forecast is issued")}

    candidates.sort(key=lambda c: -c["score"])
    for i, c in enumerate(candidates, 1):
        c["rank"] = i
    primary = candidates[0]
    runner = candidates[1] if len(candidates) > 1 else None

    # Confidence is about the PREDICTION, and it is deliberately stricter than the
    # score. Three separate things can make a prediction weak, and the first version
    # only modelled two of them:
    #
    #   signal_coverage  whether OUR data reached the signals. This is about the
    #                    plumbing, so it must NOT dominate — a user with a complete
    #                    but tiny record is not a well-understood user.
    #   sample_factor    how much ACTUAL BEHAVIOUR the record contains. A user with
    #                    3 observed actions in the window cannot have a strong habit.
    #   margin           whether the evidence separates the top action from the rest.
    #
    # Weighted accordingly: sample and margin carry more than coverage.
    actions_seen = sum(n for n in (opens, adds, reduces, closes) if n is not None)
    sample_factor = _clip(actions_seen / SAMPLE_FOR_CONFIDENCE)
    margin = primary["score"] - (runner["score"] if runner else 0.0)
    conf = (0.15 + 0.20 * primary["signal_coverage"]
            + 0.40 * sample_factor + 0.45 * margin)
    if margin < 0.05 and runner:
        conf *= 0.6
    conf = _clip(conf, 0.05, 0.9)

    return {
        "available": True,
        "version": FORECAST_VERSION,
        "window_days": win_days,
        "window": ac.get("window") or pd.get("window"),
        "horizon_days": win_days,
        "horizon_note": HORIZON_NOTE,
        "primary": primary,
        "runner_up": runner,
        "candidates": candidates,
        "confidence": round(conf, 3),
        "margin": round(margin, 3),
        "observed_actions": actions_seen,
        "sample_factor": round(sample_factor, 3),
        "confidence_note": (f"Confidence combines signal coverage ({primary['signal_coverage']:.0%}), observed action count ({actions_seen}), and the margin over the runner-up ({margin:.3f}). Thinner behavioral records reduce confidence."),
        "evidence_gaps": _gaps(pool, adds, reduces, closes, sim),
        "note": ("The forecast is derived from the user’s observed habits: viewed-but-unheld candidates, add/trim tendencies, and portfolio stability. Each action lists its signals and raw values. Missing signals reduce confidence rather than being scored as zero."),
        "caveat": ("This is a forecast of the user’s own behavioral habits, not investment advice or a market view. It describes what the user is more likely to do under the observed historical pattern, not what the user should do. Each candidate includes an observable falsification condition."),
    }


def _held_targets(held, pair_by_ticker, align, prefer=None, limit=5):
    """Held tickers as forecast targets, ordered by the relationship that fits."""
    gaps = {g.get("ticker"): g for g in (align.get("largest_gaps") or [])}
    out = []
    for t in sorted(held):
        meta = pair_by_ticker.get(t) or {}
        out.append({"ticker": t,
                    "attention_days": meta.get("attention_days"),
                    "holding_persistence": meta.get("holding_persistence"),
                    "relationship": meta.get("relationship"),
                    "position_weight": (gaps.get(t) or {}).get("position_weight")})
    if prefer == "heavy":
        # the largest positions are where a trim or a hold is most consequential
        out.sort(key=lambda x: -(x.get("position_weight") or 0))
    elif prefer == "light":
        out.sort(key=lambda x: (x.get("position_weight") is None,
                                x.get("position_weight") or 0))
    else:
        out.sort(key=lambda x: -(x.get("holding_persistence") or 0))
    return out[:limit]


def _basis(signals, claims):
    """Turn scored signals into a readable, auditable basis list.

    The claim text is written only for the signals that HAD data; a signal with no
    data is listed as a gap instead of being described as if it were measured.
    """
    out = []
    for s in signals:
        claim = claims.get(s["name"])
        if s["points"] is None:
            out.append({"name": s["name"], "weight": s["weight"], "raw": None,
                        "points": None, "claim": None,
                        "gap": "No data; this signal did not participate in scoring"})
            continue
        out.append({"name": s["name"], "weight": s["weight"], "raw": s["raw"],
                    "points": round(s["points"], 3),
                    "claim": claim or s["name"], "gap": None})
    return out


def _gaps(pool, adds, reduces, closes, sim):
    """What the forecast could NOT see, stated so a reader can discount it."""
    out = []
    if not pool:
        out.append("No viewed-but-unheld candidate pool; the new-position forecast has no target set")
    if adds is None:
        out.append("No add records; add propensity cannot be estimated")
    if reduces is None and closes is None:
        out.append("No trim or exit records; exit propensity cannot be estimated")
    if sim is None:
        out.append("No day-to-day holding-set similarity; portfolio stability cannot be estimated")
    return out


def digest(forecast):
    """Compact view for prompts."""
    f = forecast or {}
    if not f.get("available"):
        return {"available": False, "reason": f.get("reason")}
    p = f.get("primary") or {}
    return {"available": True, "version": f.get("version"),
            "primary": p.get("action"), "primary_zh": p.get("action_zh"),
            "primary_score": p.get("score"), "confidence": f.get("confidence"),
            "runner_up": (f.get("runner_up") or {}).get("action"),
            "targets": [t.get("ticker") for t in (p.get("targets") or [])][:5],
            "candidates": [{c["action"]: c["score"]} for c in f.get("candidates") or []],
            "caveat": f.get("caveat")}
