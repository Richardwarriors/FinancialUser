"""B-class behaviour observation signals.

A separate observation tree, deliberately NOT part of the 7 KYC L1 / 60 L3
feature count: these describe how the user *behaves on the platform*, whereas
the KYC tree describes what we know about their situation. Both are grounded in
real data; these are all computed, so they never need a questionnaire to fill.

Two of the four signals were named explicitly in the source spec
(`research_depth_signal`, `advice_acceptance_pattern`); the other two are
derived from what the BM streams can actually support, and are flagged as such.

Honesty constraints that shaped this module:

* BM428 conversation titles carry **no timestamps**. So "advice acceptance"
  cannot be measured as "was advised -> then acted" — that ordering is not in
  the data. We measure advice-*seeking* behaviour and state the limitation
  rather than implying a causal chain.
* Interaction counts are not interest, and interest is not conviction: every
  label here is a description of observed platform behaviour, nothing more.
"""
from collections import Counter
from datetime import timezone
import re

from .behavior import load_weights

# ---------- conversation intent classification ----------

_CONV_PATTERNS = {
    "price_target": re.compile(
        r"(到什么价位|能到多少|目标价|预期.{0,6}(价位|价格|点位)|预计.{0,6}(价位|价格)|"
        r"涨跌幅|盘后|盘前|合理估值|多少钱|price target|how high|how low)", re.I),
    "deep_analysis": re.compile(
        r"(深入分析|分析一下|对比|对比一下|拆解|评估一下|怎么看|前景|"
        r"deep analysis|breakdown|compare|outlook|thesis)", re.I),
    "account_inquiry": re.compile(
        r"(我的账户|我的持仓|我的现金|我有多少|账户现状|账户是什么情况|"
        r"我买(了)?什么|持仓情况|my (account|portfolio|positions|holdings))", re.I),
    "self_awareness": re.compile(
        r"(你对我了解多少|你对我的kyc|对我的了解|你知道我|讲出来|了解我|"
        r"what do you know about me|my profile)", re.I),
    "format_request": re.compile(
        r"(表格|列表|列出来|整理|导出|汇总|按.{0,4}呈现|分点|"
        r"table|summari[sz]e|export|in a list)", re.I),
    "news_event": re.compile(
        r"(财报|业绩|局势|冲突|非农|议息|加息|降息|事件|earning|conflict|fed|tariff)", re.I),
    "comparison_peer": re.compile(r"(哪一家|哪个更好|比较|vs\.?|相比|对比.{0,6}哪)", re.I),
}

# The patterns above match the SOURCE titles, which are Chinese, so they stay as
# they are. These are the OUTPUT names for the same buckets: this block is written
# into the published artefacts, and the English boundary can only strip prose it
# cannot translate, so the English has to be written here.
_CONV_LABEL = {
    "price_target": "asked for a price target directly",
    "deep_analysis": "asked for deep analysis",
    "account_inquiry": "asked about their own account",
    "self_awareness": "asked what the system knows about them",
    "format_request": "asked for a specific output format",
    "news_event": "asked about a news event",
    "comparison_peer": "asked for a peer comparison",
}


def classify_conversations(snap):
    """Bucket BM428 titles by what the user is asking for."""
    titles = [str(x.get("title") or "").strip() for x in (snap.get("conversations_1m") or [])]
    titles = [t for t in titles if t]
    counts = Counter()
    for t in titles:
        for kind, rx in _CONV_PATTERNS.items():
            if rx.search(t):
                counts[kind] += 1
    return titles, counts


# ---------- signal builders ----------

def research_depth_signal(snap, intent, weights_state):
    """How deep does the research actually go (BM433 structure, not volume)."""
    if not intent or not intent.get("available"):
        return None
    rd = intent.get("research_depth") or {}
    rb = intent.get("research_breadth") or {}
    style = intent.get("research_style") or {}
    sessions = rd.get("sessions") or 0
    grade = rd.get("research_grade_session_share")
    mins = rd.get("median_session_minutes")
    tech = style.get("technical_driven") or 0
    fund = style.get("fundamental_driven") or 0
    price = style.get("price_check_share") or 0

    if not sessions:
        return {"value": "insufficient_data", "status": "gap",
                "display": "No identifiable research session, so research depth "
                           "cannot be assessed.",
                "evidence": [], "confidence": 0.2,
                "data_needed": ["Denser behavioural instrumentation is required"]}

    if grade is not None and grade >= 0.5 and fund >= 0.05:
        label = "deep_research"
        note = ("A substantial share of research sessions reaches fundamentals or "
                "earnings level rather than stopping at quotes")
    elif grade is not None and grade >= 0.5:
        label = "structured_technical_research"
        note = ("Research is organised and switches timeframes and tools, but stays "
                "on price and technicals with very little fundamental content")
    elif grade is not None and grade >= 0.2:
        label = "moderately_shallow"
        note = ("There is some research activity, but most sessions still stop at a "
                "quick look")
    else:
        label = "surface_browsing"
        note = ("Behaviour is dominated by quick checks and rarely forms a "
                "continuous research process")

    return {
        "value": label,
        "status": "filled",
        "display": f"{note}. {sessions} research sessions, {grade:.0%} of them "
                   f"research-grade, median session length {mins:g} min; behaviour "
                   f"mix: quote checks {price:.0%}, technicals {tech:.0%}, "
                   f"fundamentals {fund:.0%}.",
        "evidence": [f"behavior_intent.research_depth.research_grade_session_share={grade}",
                     f"behavior_intent.research_depth.sessions={sessions}",
                     f"behavior_intent.research_style.fundamental_driven={fund}"],
        "confidence": round(min(0.5 + grade * 0.4, 0.9), 2),
        "coverage": "deterministic", "method": "rule",
    }


def attention_persistence_signal(snap, intent, weights_state=None):
    """One-off curiosity vs sustained tracking (BM433 time spread)."""
    if not intent or not intent.get("available"):
        return None
    tot = intent.get("totals") or {}
    days = tot.get("active_days") or 0
    dr = tot.get("date_range") or [None, None]
    att = intent.get("attention") or []
    themes = intent.get("themes") or []

    if not att:
        return {"value": "insufficient_data", "status": "gap",
                "display": "No behaviour can be attributed to a specific ticker, so "
                           "attention persistence cannot be assessed.",
                "evidence": [], "confidence": 0.2,
                "data_needed": ["Ticker-level behavioural data is required"]}

    # a ticker counts as "tracked" when it shows up on many distinct days
    tracked = [a for a in att if (a.get("active_days") or 0) >= 10]
    multi_week = [a for a in att if (a.get("active_days") or 0) >= 5]
    rising = [t for t in themes if t.get("trend") == "increasing"]
    ratio = len(tracked) / len(att)

    if ratio >= 0.5:
        label = "sustained_tracking"
        note = ("Most watched tickers are revisited across many days, which is "
                "sustained rather than one-off attention")
    elif multi_week:
        label = "episodic_tracking"
        note = ("Some tickers are tracked for a stretch of days while the rest draw "
                "only short-lived interest")
    else:
        label = "short_burst_attention"
        note = ("Attention clusters into a few sessions with no multi-day "
                "follow-through")

    return {
        "value": label,
        "status": "filled",
        "display": f"{note}. Observation window {days} days ({dr[0]} to {dr[1]}); "
                   f"{len(tracked)} of {len(att)} frequently viewed tickers appear "
                   f"on 10 or more days; "
                   + (f"{len(rising)} themes are drawing rising attention."
                      if rising else "No theme shows a clear rise in attention."),
        "evidence": [f"behavior_intent.totals.active_days={days}",
                     f"behavior_intent.attention[].active_days "
                     f"({len(tracked)} tickers >= 10 days)"],
        "confidence": 0.75, "coverage": "deterministic", "method": "rule",
    }


def execution_discipline_signal(snap, intent, weights_state=None):
    """Does research precede execution, and is the book handled consistently?"""
    if not intent or not intent.get("available"):
        return None
    b2t = intent.get("behavior_to_trade") or {}
    conv = intent.get("attention_conversion") or {}
    trades = snap.get("trades_6m") or []
    if not b2t.get("available"):
        return {"value": "insufficient_data", "status": "gap",
                "display": "No executed trades, so the relationship between "
                           "research and execution cannot be assessed.",
                "evidence": [], "confidence": 0.2,
                "data_needed": ["Execution data (BM452) is required"]}

    prior = b2t.get("prior_research_rate")
    lat = (b2t.get("research_to_trade_latency_hours") or {}).get("median")
    deep = conv.get("deep_research_to_trade_conversion")
    view = conv.get("view_to_trade_conversion")
    n = b2t.get("trades_total") or 0

    if prior is not None and prior >= 0.6 and lat is not None and lat >= 12:
        label = "research_then_execute"
        note = ("Most fills are preceded by traceable research, with deliberation "
                "time left between the research and the order")
    elif prior is not None and prior >= 0.6:
        label = "researched_but_fast_execution"
        note = ("Fills are generally preceded by research, but the gap between "
                "research and order is short, which points to fast decisions")
    elif prior is not None and prior >= 0.2:
        label = "partly_unresearched"
        note = ("Only some fills have research in front of them; the rest look "
                "closer to impromptu trades")
    else:
        label = "no_traceable_research"
        note = ("Almost no traceable research precedes a fill: the research may "
                "happen elsewhere, or the decision is already formed")

    parts = [f"{n} fills, {prior:.0%} of them with same-ticker research in the "
             f"preceding 7 days" if prior is not None else f"{n} fills"]
    if lat is not None:
        parts.append(f"median research-to-order gap {lat / 24:.1f} days")
    if deep is not None and view is not None:
        parts.append(f"conversion after deep research is {deep:.1%}, clearly above "
                     f"the {view:.1%} that follows a plain quote check"
                     if deep > view else
                     f"conversion is {view:.1%} after a quote check and {deep:.1%} "
                     f"after deep research")

    return {
        "value": label, "status": "filled",
        "display": note + ". " + "; ".join(parts) + ".",
        "evidence": [f"behavior_intent.behavior_to_trade.prior_research_rate={prior}",
                     f"behavior_intent.behavior_to_trade.research_to_trade_latency_hours.median={lat}",
                     f"behavior_intent.attention_conversion.deep_research_to_trade_conversion={deep}"],
        "confidence": 0.8, "coverage": "deterministic", "method": "rule",
    }


def advice_acceptance_pattern(snap, intent, weights_state=None):
    """Advice-seeking behaviour from BM428.

    Named `advice_acceptance` in the source spec, but the stream has no
    timestamps, so acceptance (advice -> subsequent action) is NOT observable.
    We therefore report the seeking side and say so plainly.
    """
    titles, counts = classify_conversations(snap)
    if not titles:
        return {"value": "insufficient_data", "status": "gap",
                "display": "No conversation record, so advice-seeking behaviour "
                           "cannot be observed.",
                "evidence": [], "confidence": 0.2,
                "data_needed": ["The BM428 conversation stream is required"]}

    n = len(titles)
    asked = sum(counts.values())
    account = counts.get("account_inquiry", 0) + counts.get("self_awareness", 0)
    direct = counts.get("price_target", 0)
    deep = counts.get("deep_analysis", 0) + counts.get("comparison_peer", 0)

    if account / n >= 0.2:
        label = "verifies_own_situation"
        note = ("Actively asks about their own account and about what the system "
                "knows, so willingness to consult is high")
    elif (direct + deep) / n >= 0.4:
        label = "answer_seeking"
        note = ("Questions mostly ask for a concrete conclusion -- a price target, "
                "a comparison, an analysis -- which is outcome-oriented")
    elif deep / n >= 0.2:
        label = "research_oriented_questions"
        note = ("Questions centre on analysis and comparison, which suggests the "
                "user wants the material and will judge for themselves")
    else:
        label = "little_advisory_interaction"
        note = ("The conversation record is limited and shows no recognisable "
                "advisory pattern")

    mix = ", ".join(f"{_CONV_LABEL[k]} x{v}" for k, v in counts.most_common(4)) \
        or "no identifiable category"
    return {
        "value": label, "status": "filled",
        "display": f"{note}. {n} conversations: {mix}. Note: BM428 titles carry no "
                   f"timestamps, so whether advice was adopted cannot be aligned "
                   f"with later trades; this describes advice-seeking behaviour "
                   f"only, not acceptance.",
        "evidence": [f"bm428_conversations total {n}, {asked} matched a category",
                     f"bm428 account questions {account}, direct price-target "
                     f"questions {direct}, analysis questions {deep}"],
        "confidence": 0.7 if n >= 10 else 0.5,
        "coverage": "deterministic", "method": "rule",
        "limitation": ("Conversations carry no timestamps, so advice adoption "
                       "cannot be verified"),
    }


# ---------- assembly ----------

SIGNALS = [
    ("research_depth_signal", "Research Depth Signal",
     "Depth of research effort shown by behavioural evidence (quote watching vs "
     "fundamentals/earnings-level work)",
     ["BM433"]),
    ("attention_persistence_signal", "Attention Persistence Signal",
     "Whether attention is one-off or tracked across days", ["BM433"]),
    ("advice_acceptance_pattern", "Advice Acceptance Pattern",
     "How the user asks for advice in conversation (note: no timestamps, so "
     "adoption cannot be determined)", ["BM428"]),
    ("execution_discipline_signal", "Execution Discipline Signal",
     "Whether research precedes trading, and the research-to-order interval",
     ["BM433", "BM452", "BM453"]),
]

BUILDERS = {
    "research_depth_signal": research_depth_signal,
    "attention_persistence_signal": attention_persistence_signal,
    "advice_acceptance_pattern": advice_acceptance_pattern,
    "execution_discipline_signal": execution_discipline_signal,
}


def build_behavior_signals(snap, intent=None, evidence=None):
    """B-class block: {"signals": {...}, "summary": {...}}"""
    weights_state = load_weights()
    out, filled = {}, 0
    for key, label, desc, streams in SIGNALS:
        try:
            r = BUILDERS[key](snap, intent, weights_state)
        except Exception as e:                                        # noqa: BLE001
            r = {"value": "insufficient_data", "status": "gap",
                 "display": f"Computation failed: {type(e).__name__}",
                 "evidence": [], "confidence": 0.0,
                 "data_needed": ["Re-run this signal"]}
        if r is None:
            r = {"value": "insufficient_data", "status": "gap",
                 "display": "Current data is insufficient to compute this signal.",
                 "evidence": [],
                 "confidence": 0.0, "data_needed": streams}
        r = {**r, "key": key, "label": label, "description": desc, "data_sources": streams,
             "source": "computed" if r.get("status") == "filled" else "gap"}
        out[key] = r
        if r["status"] == "filled":
            filled += 1
    result = {
        "count": len(SIGNALS),
        "filled": filled,
        "note": ("B-class signals observe platform behaviour. They are deliberately "
                 "outside the KYC 7 L1 / 60 L3 feature count, are computed directly "
                 "from BM data without a questionnaire, and do not replace a KYC "
                 "conclusion."),
        "signals": out,
    }
    from .english_output import englishize
    return englishize(result)
