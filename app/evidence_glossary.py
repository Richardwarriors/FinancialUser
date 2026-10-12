"""Human-readable explanation of the evidence cited under every fact.

Why this module exists
----------------------
The tree cites its evidence as field paths, because that is what makes a claim
auditable:

    bm453_daily[2026-09-06].holdings 按 instrument 分类聚合（7 个标的）
    bm453_daily.asset_total=1291671.592
    behavior_intent.research_depth.sessions=374
    active_days
    view_company

Read by a person, none of these says what it IS. `按 instrument 分类聚合` is
internal vocabulary; `active_days` is not even a sentence; and `asset_total` is
the single most misreadable field in the whole product, because it INCLUDES
deposits and withdrawals, so a reader who takes it for a return figure draws a
wrong conclusion from a correct number.

So the citation stays (it is the audit trail) and a plain reading is added beside
it: what the source is, what the field measures, what the value says, and — where
it matters — what it does NOT establish. `annotate()` returns both, and the viewer
leads with the sentence.

Honesty rules, same as everywhere else in this project:

* **Never invent a value.** If the line carries no `=`, the reading describes the
  field without asserting a number.
* **An unknown field is declared unknown**, not guessed at. `explain()` reports
  `known: False` and the reading says the glossary has no entry, rather than
  dressing an unfamiliar token up as an explanation.
* **A caveat travels with the field, not with the call site**, so `asset_total`'s
  deposit warning cannot be forgotten by a future producer.
"""
import json
import re

EVIDENCE_GLOSSARY_VERSION = "evidence-glossary-v1"

# The data streams. Named because "bm453_daily" is an internal identifier the
# reader has never seen, and which stream a fact came from changes how much it
# should be trusted.
SOURCES = {
    "bm433": ("BM433 behavior event stream",
              "In-app behavioral tracking: viewing quotes, switching periods, reading news, viewing company information, etc., reflects what the user viewed, not their actual fund movements."),
    "bm452_trades": ("BM452 transaction records",
                     "Actual transaction records (transaction time, price, and quantity), the primary basis for \"what was actually done.\""),
    "bm453_daily": ("BM453 daily account snapshots",
                    "One record per trading day: total assets, securities market value, cash, and per-instrument position details (quantity, cost price, latest price, market value, unrealized P&L). The foundation for all fund-related conclusions."),
    "bm428_conversations": ("BM428 conversation records",
                            "Conversation titles between the user and the AI assistant. Titles only, with no timestamps, so it cannot be determined whether \"a given recommendation was adopted\"."),
    "behavior_intent": ("Behavior Intent Engine",
                        "Deterministic derivations from BM433 events: session segmentation, research depth, attention weights, and the correspondence between behavior and transactions."),
    "behavior": ("Behavior Layer", "Statistical basis for BM433 behavior events."),
    "bse": ("Behavior Signal Engine", "Observation signals derived from behavioral events."),
    "position": ("Position dynamics engine",
                 "Inferred from BM453 daily positions: start a new position/add/trim/exit events, profit and loss status, averaging down, and profit realization behavior."),
    "portfolio_metrics": ("Portfolio Metrics Engine", "Return and risk metrics based on a time-weighted net asset value series."),
    "trade_metrics": ("Trade metrics engine", "Holding periods and win/loss statistics based on FIFO matching."),
    "attention_capital": ("Attention-funds cross-engine", "Compare behavioral attention with actual capital allocation."),
    "computed_portfolio_metrics": ("Portfolio Metrics Engine",
                                   "Asset and investable amount ranges calculated from the account snapshot."),
    "intent_funnel": ("Intent funnel", "Counts at each stage: attention→research→conviction→trading."),
    "info_type_mix": ("Information type composition", "The number and proportion of each type of information (news/fundamentals/technical)."),
    "kyc": ("KYC feature layer", "Features and gaps organized by the KYC tag tree."),
    "asset": ("Asset evidence layer", "The net asset value curve and its derived returns/drawdown/volatility."),
}

# field path -> (中文名, 含义, 局限). The limitation travels with the field.
FIELDS = {
    # ---- BM453 账户 ---------------------------------------------------------
    "asset_total": ("Total account assets", "The total securities market value and cash on that day.",
                    "Includes deposits and withdrawals. An increase does not mean money was earned and cannot be read as a rate of return; the rate of return must use time-weighted net asset value (with deposits and withdrawals removed)."),
    "stock_total": ("Securities market value", "The total market value of instruments held on that day, excluding cash."),
    "money_remain": ("Cash balance", "Cash not invested in securities on that day."),
    "snapshot_date": ("Snapshot date", "The trading day corresponding to this position snapshot."),
    "holdings": ("Holding details by instrument",
                 "The quantity, cost basis, latest price, market value, and unrealized profit/loss for each instrument on that day.",
                 "Covers only accounts visible on this platform; external assets are unobservable."),
    "profit_loss_total": ("Unrealized P&L", "Position unrealized profit/loss relative to cost.",
                          "There is a systematic deviation from the cost-price basis; determine profit/loss status exclusively using \"latest price vs cost price\"."),
    "stock_market_value": ("Position market value", "The instrument's position quantity × latest price."),
    "cost_price": ("Cost price", "The weighted average purchase cost of the instrument's position."),
    "last_price": ("Latest price", "The closing price of this instrument on the snapshot date."),

    # ---- 行为引擎 -----------------------------------------------------------
    "active_days": ("Active days", "Number of days with at least one behavioral event.",
                    "\"Opening the App\" does not equal \"doing research\"; a high number of active days only indicates frequent use."),
    "events": ("Number of behavioral events", "Total number of behavioral events included in the statistics."),
    "event_count": ("Number of behavioral events", "Total number of behavioral events included in the statistics."),
    "avg_events_per_active_day": ("Average daily behavioral event count",
                                  "Number of behavioral events ÷ active days, measuring how much was done after opening the app."),
    "avg_top1_concentration": ("Mean maximum single-instrument share",
                               "Daily average of \"largest single holding ÷ securities market value\"."),
    "event_mix": ("Behavioral event composition", "The number and proportion of each type of behavioral event."),
    "event_type": ("Behavioral event type", "The specific type of this action (checking prices/viewing news/viewing options, etc.)."),
    "info_type": ("Information type", "The type of information to which the behavior belongs (news/fundamentals/technical/options/social)."),
    "view_earnings": ("Earnings view count", "Number of times instrument earnings pages were opened."),
    "research_depth": ("Research depth", "Session count, duration, event density, and the proportion of \"research-grade\" sessions."),
    "research_breadth": ("Research breadth", "Number of covered instruments, attention concentration (HHI), and maximum share."),
    "research_style": ("Browsing information type composition",
                       "Share of browsing events by information type (news/fundamentals/technical/options/social/price checking).",
                       "This is \"what was viewed,\" not \"how deeply it was analyzed,\" and cannot be treated as research quality."),
    "sessions_top": ("Primary research session", "Several sessions with the most events: instrument, date, duration, and peak stage."),
    "sessions": ("Number of research sessions", "Number of segmented research sessions."),
    "research_grade_sessions": ("Number of research-grade sessions", "Number of sessions reaching the \"research-grade\" depth threshold."),
    "research_grade_session_share": ("Share of research-grade sessions", "Research-grade sessions ÷ all sessions."),
    "median_session_events": ("Events per session (median)", "Median number of behavioral events per session."),
    "median_session_minutes": ("Session duration (median)", "Median session duration in minutes."),
    "attention_hhi": ("Attention concentration HHI", "Sum of squared attention proportions across instruments. Lower means more diversified."),
    "top_ticker_share": ("Maximum single-instrument attention share", "Proportion of total attention accounted for by the most-followed instrument."),
    "top_theme_share": ("Maximum single-theme share", "Share of all tracked interests represented by the most concentrated theme."),
    "unique_tickers": ("Number of instruments covered by attention", "Number of distinct instruments observed during the observation window."),
    "timeframe_switches": ("Number of cycle switches", "Number of times the time period was switched on charts; classified as price-checking behavior."),
    "view_company": ("Number of company information views", "Number of times fundamental information for an instrument's company was opened."),
    "view_diagnosis": ("Diagnostic view count", "Number of times individual stock diagnostic/rating pages were opened."),
    "view_options": ("Number of options page views", "Number of times option-related pages were opened."),
    "view_quote": ("Quote view count", "Number of times an instrument quote page was opened."),
    "read_news": ("Number of news reads", "Number of times news content was opened."),
    "switch_timeframe": ("Number of switching cycles", "Switching the time period on charts; classified as price-checking behavior."),
    "news_driven": ("News-driven share", "Proportion of browsing events involving news."),
    "fundamental_driven": ("Fundamental-driven proportion", "Proportion of browsing events involving company fundamentals."),
    "technical_driven": ("Technical analysis-driven share", "Proportion of browsing events involving technical charts/indicators."),
    "options_driven": ("Options-driven share", "Proportion of browsing events involving options-related content."),
    "social_driven": ("Social-driven proportion", "Proportion of browsing events involving community/social content."),
    "price_check_share": ("Quote lookup share", "Proportion of browsing events involving price-only viewing."),
    "prior_research_rate": ("Proportion of trades with research before order placement",
                            "Proportion of trades with research occurring within 48 hours before the trade.",
                            "Can only establish sequence of events, not that research caused this trade."),
    "research_to_trade_latency_hours": ("Lag from research to execution",
                                        "The number of hours between research activity and actual execution."),
    "deep_research_to_trade_conversion": ("Deep Research-to-Trade Conversion Rate",
                                          "Proportion of instruments reaching deep research that were ultimately purchased."),
    "attention_conversion": ("Attention conversion", "Conversion along the attention→research→transaction path."),
    "behavior_to_trade": ("Behavior-to-trade correspondence", "Whether behavioral events correspond to actual trades."),
    "totals": ("Total Behavior", "Total behavioral events and active days during the observation window."),
    "attention": ("Instrument attention", "Number of attention days and behavior composition by instrument."),

    # ---- 持仓动态 -----------------------------------------------------------
    "max_num_positions": ("Maximum simultaneous positions", "Maximum number of instruments held simultaneously during the observation window."),
    "avg_num_positions": ("Average number of holdings", "Daily average number of holdings."),
    "entries": ("Position-start event", "Observable new positions during the observation window.",
                "Instruments already held at the start of the observation window cannot be used to determine the entry time and are excluded."),
    "exits": ("Exit Event", "Fully exited an instrument during the observation window."),
    "scale_ups": ("Add-position event", "Continue buying an existing position, counted only when the change relative to the previous day reaches the threshold."),
    "scale_downs": ("Trim event", "Partially sell an existing position."),
    "events_total": ("Total events in observation window", "The total number of events actually included in the calculation for this metric family."),

    # ---- 关注-资金交叉 -------------------------------------------------------
    "loss_attention_ratio": ("Attention ratio (loss/profit)",
                             "Attention volume during unrealized losses ÷ attention volume during unrealized gains. Greater than 1 indicates more viewing during unrealized losses.",
                             "Only shows that \"attention\" and an unrealized loss occurred simultaneously; it cannot prove causation."),
    "share_on_losing": ("Unrealized-loss attention share", "Proportion of attributable attention events that occurred during an unrealized loss."),
    "losing": ("Attention count under unrealized losses", "Number of attention events while in an unrealized loss."),
    "winning": ("Attention Count During Unrealized Profit", "Number of attention events while in an unrealized gain."),
    "dominant_state": ("Primary relationship", "Primary combination pattern between attention and holdings."),
    "attention_gt_capital": ("Instruments with more attention than allocation",
                             "Viewed frequently but with a low proportion of invested funds; candidate pool of \"viewed but not yet purchased.\""),
    "capital_gt_attention": ("Allocation exceeds attention",
                             "High position proportion but little attention usually indicates passive or legacy holdings."),
    "holding_persistence": ("Holding persistence", "The proportion of days during the observation window on which this instrument was held."),

    # ---- 组合 / 资产 ---------------------------------------------------------
    "avg_top1_share": ("Average maximum single-instrument share", "Daily average of \"largest single holding ÷ securities market value\"."),
    "top1_share": ("Maximum single-instrument share", "The proportion of securities market value represented by the largest single holding on that day."),
    "top_sector_share": ("Maximum sector share", "Share of securities market value held by the industry with the highest proportion."),
    "sector_map_coverage": ("Industry table coverage", "Share of market capitalization identifiable from the static industry table.",
                            "When the coverage rate is low, industry concentration is underestimated."),
    "curve_coverage": ("Net asset value curve availability", "Proportion of credible days in the net asset value curve."),
    "best_day_return": ("Best daily return", "The best day in the net asset value series."),
    "equity_change_abs": ("Change in net asset value", "Absolute amount of change in equity during the observation window."),
    "max_drawdown": ("Maximum drawdown", "The maximum decline from the historical high to the lowest point."),
    "avg_equity": ("Average equity", "The average level of the net asset value series."),

    # ---- 对话 ---------------------------------------------------------------
    "chinese_share": ("Proportion of Chinese-language conversations", "Proportion of Chinese in conversation titles."),
    "conversation": ("Conversation classification", "Categorization and counting of conversation titles by topic."),

    # ---- 其余引用中出现的字段 ---------------------------------------------
    "total_assets": ("Total account assets", "The total securities market value and cash at that point in time.",
                     "Includes deposits and withdrawals; cannot be read as a rate of return."),
    "latest_asset_total": ("Latest total account assets", "Total assets on the most recent snapshot date.",
                           "Includes deposits and withdrawals; cannot be read as a rate of return."),
    "latest_share": ("Latest proportion", "The proportion of securities market value represented by this instrument in the latest snapshot."),
    "single_name_equity": ("Individual stock proportion", "Share of a single company's stock in securities market value."),
    "band": ("Investable amount range", "Estimated investable amount range based on the account balance."),
    "short_timeframe_share": ("Short-cycle viewing proportion",
                              "Proportion of short observation windows (minute/hour level) among the time periods selected for browsing."),
    "timeframe_switch_count": ("Number of cycle switches", "Number of times the time period was switched on charts."),
    "evaluation": ("Assessment stage", "Number of events in the \"judgment/comparison\" stage of the intent funnel."),
    "conviction": ("Conviction stage", "Number of events in the \"conviction\" stage of the intent funnel."),

    # Generic statistic keys, used inside the event/scale-up citations.
    "count": ("Quantity", "The number of occurrences of the event or action group."),
    "change": ("Magnitude of change", "Proportion of the change in position market value relative to the previous day."),
    "state_before": ("Profit/loss status before change", "The instrument's profit/loss status relative to its cost basis one day before the action occurred."),
    "state_during": ("Period P&L status", "The predominant profit/loss state during the group's continuous actions."),
    "state_after": ("Profit/loss status after change", "The instrument's profit/loss status relative to its cost basis after the action occurred."),
    "weight": ("Position proportion", "The proportion of total holdings market value represented by this instrument."),
    "index": ("Day number within the window", "The number of days from the first day of the observation window to the action, used to calculate holding days."),
    "action": ("Action type", "Start a new position / add / trim / exit."),
    "open": ("Start a new position", "Establishing a position from zero."),
    "add": ("Add", "Continue buying an existing position."),
    "reduce": ("Trim", "Partially sell an existing position."),
    "close": ("Exit", "Sell all of an instrument."),
    "reenter": ("Start a new position", "Buy the Same Instrument Again After Exiting."),

    # Info-type sub-counts: `info_type_mix.fundamental.count=39`.
    "fundamental": ("Fundamental category", "Behavior/events from company fundamental information."),
    "technical": ("Technical analysis category", "Behavior/events from technical charts and indicators."),
    "news": ("News-related", "Behavior/events from news content."),
    "options": ("Options", "Behavior/events from options-related content."),
    "social": ("Social category", "Behavior/events from community/social content."),
    "price": ("Quote lookup category", "Only observes price-related behavior/events."),
    "research": ("Research stage", "Number of events in the \"research\" stage of the intent funnel."),
}

# Which stream a prefix belongs to, longest first so `bm452_trades` is not read
# as `bm452`.
_PREFIXES = sorted(SOURCES, key=len, reverse=True)

# `asset_total=1291671.592` -> path `asset_total`, value in the tail. The latin
# field path is simply the LEADING latin run: a CJK qualifier, `=`, or whitespace all
# end it. A trailing lookahead on `\s|=|$` looked right and silently failed whenever a
# path was followed straight by prose (`attention[].active_days（8 个标的）`).
_PATH = re.compile(r"^([A-Za-z_][A-Za-z0-9_.\[\]]*)")
_BRACKET = re.compile(r"^\[([^\]]*)\]")
# Already-written prose: contains Chinese, or is a sentence.
_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

# Suffixes that are counters, not the field itself.
_COUNTER_SUFFIX = {"median", "max", "min", "mean", "p25", "p75", "share", "rate",
                   "ratio", "count", "total", "avg", "sum", "days", "hours"}


# Producer qualifiers, in the internal vocabulary the pipeline emits. Each one is
# a fragment of a sentence that a reader cannot act on: `按 instrument 分类聚合`
# names an internal enum, `按发行人属地映射` names a mapping step. Translating them
# here is what makes the citation about the holdings table readable, which was the
# reported problem.
PHRASES = {
    "Aggregate by instrument category": "Aggregate after classifying by instrument type (individual stocks / thematic ETFs / broad-market ETFs / leveraged inverse ETFs / fixed income and cash / commodities)",
    "Map by issuer domicile": "Classify by the instrument's place of issuance (used to determine whether holdings are concentrated in a single market)",
    "Map using the static industry table": "Classify according to the built-in industry reference table (underestimated concentration indicates insufficient table coverage)",
    "By place of issuance": "Classify by the instrument's place of issuance",
    "Daily top1 market value proportion": "Calculate the proportion of the largest single holding to securities market value daily",
    "top1": "Largest single holding by proportion",
    "Market value share": "Proportion of securities market value",
    "Coverage Rate": "Identifiable proportion",
    "Non-U.S. instruments": "Instruments not registered in the United States",
    "Leveraged product trades": "Trades buying leveraged/inverse products",
    "Leveraged/inverse products": "Leveraged or inverse ETFs",
    "External deposits and withdrawals": "Funds transferred in or out externally (not investment returns)",
    "Total securities market value": "Total market value of all holdings",
}


def _phrase(text):
    """Replace internal qualifier phrases with plain wording, longest first."""
    out = text
    for k in sorted(PHRASES, key=len, reverse=True):
        if k in out:
            out = out.replace(k, PHRASES[k])
    return out


def _split_source(text):
    """`bm453_daily[2026-09-06].holdings 共 7 个` -> ("bm453_daily", rest).

    Returns `(None, text)` when the head is not a known stream. Note that a known
    FIELD is also not a source: `active_days` names a field, not a data stream.
    """
    for p in _PREFIXES:
        if text == p or text.startswith(p + ".") or text.startswith(p + "[") \
                or text.startswith(p + " "):
            return p, text[len(p):]
    return None, text


def _lookup(path):
    """Glossary entry for a dotted path, falling back to its longest known head.

    `research_depth.sessions` resolves via `sessions`; `attention[].active_days`
    resolves via `active_days`. A suffix like `median` is a statistic rather than
    a field, so it is skipped rather than reported as an unknown field.
    """
    parts = [p for p in re.split(r"[.\[]+", path) if p and p != "]" and not p.isdigit()]
    for part in reversed(parts):
        if part in _COUNTER_SUFFIX and len(parts) > 1:
            continue
        if part in FIELDS:
            return part, FIELDS[part]
    return None, None


# Enum values live in ENUM_VALUES (further down) and are applied by `_enum`. There is
# deliberately only ONE table. A second, key-gated copy used to live here and
# translated a different subset, so `dominant_state=minor_holding` stayed opaque while
# `event_type=read_news` was translated in the same sentence — half-readable, which is
# worse than uniformly opaque because it looks finished.


# Python repr signatures. Deliberately narrow: a bare bracket or brace is
# legitimate punctuation in Chinese prose, so only quoted-container forms qualify.
# Lives HERE rather than in profile_tree so there is one implementation: the repr
# leak shipped twice from two producers, and a second copy is how a third happens.
_REPR_MARKERS = ("{'", "[{", "', '", "': ", "'}", 'object at 0x')
_PY_DICT = re.compile(r"\{([^{}]*)\}")
_PY_LIST = re.compile(r"\[([^\[\]]*)\]")


def de_repr(text):
    """Render Python-repr fragments inside a string as readable text.

    A last line of defence at the presentation boundary. The producers are fixed,
    but a stored profile can predate the fix, and a new upstream source can
    reintroduce the pattern — the leak has already shipped twice from two different
    modules. Converting the common shapes here means the reader is correct either way:

        杠杆产品成交={'SOXL': 8}          -> 杠杆产品成交=SOXL×8
        attention_gt_capital=['OKLO','GH'] -> attention_gt_capital=OKLO、GH

    Only strings that actually look like a repr are touched, so ordinary prose is
    returned unchanged.
    """
    if not isinstance(text, str) or not any(m in text for m in _REPR_MARKERS):
        return text

    def _dict(m):
        parts = []
        for pair in m.group(1).split(","):
            k, _, v = pair.partition(":")
            k, v = k.strip().strip("'\""), v.strip().strip("'\"")
            if k:
                parts.append(f"{k}×{v}" if v else k)
        return "、".join(parts) or m.group(0)

    def _list(m):
        items = [x.strip().strip("'\" ") for x in m.group(1).split(",")]
        return "、".join(x for x in items if x) or m.group(0)

    # Mappings first: a list of mappings then collapses as a plain list.
    return _PY_LIST.sub(_list, _PY_DICT.sub(_dict, text))


# Values that are INTERNAL CODES. Showing `dominant_state=minor_holding` to a
# reader is the clearest form of the "standardised evidence" problem: the code is
# consistent, machine-checkable, and completely opaque. The code stays in the raw
# citation (it is what the pipeline compares on) and the Chinese reading is shown
# beside it. Measured fleet-wide: the alignment vocabulary alone appears 147 times.
ENUM_VALUES = {
    # attention_capital 2x2 alignment vocabulary
    "high_conviction_monitoring": "Key focus (high attention, also heavily held)",
    "potential_entry": "Candidate position (high attention but light holding)",
    "passive_core_holding": "Passive Core Holding (heavy position, little attention)",
    "minor_holding": "Small position sizes (few interests and holdings)",
    # intent stages
    "attention": "Attention",
    "research": "Research",
    "evaluation": "Assessment",
    "conviction": "Conviction",
    "execution": "Trading",
    "holding": "Hold",
    # risk
    "conservative": "Conservative",
    "moderate": "Moderate",
    "aggressive": "Aggressive",
    "very_aggressive": "Very aggressive",
    "unknown": "Unknown",
    # horizon / style
    "short": "Short term",
    "medium": "Medium term",
    "long": "Long term",
    "mixed": "Mixed",
    "long_term_investor": "Long-term investor",
    "swing_trader": "Swing trader",
    "active_trader": "Active trader",
    "high_frequency": "High-frequency trading",
    "observer_only": "Observes only, does not trade",
    # P&L / event state
    "loss": "Unrealized loss",
    "gain": "Unrealized gain",
    "flat": "Flat",
    # behavior event types.
    # Wording matches the FIELD labels above on purpose: a reader who sees
    # 「阅读新闻次数」 as a field name must not see 「读新闻」 for the same event
    # elsewhere, or the two read as different things.
    "view_quote": "View quote",
    "view_news_page": "Browse news pages",
    "read_news": "Read news",
    "view_company": "View company information",
    "view_earnings": "View earnings",
    "view_options": "View options page",
    "switch_timeframe": "Switching cycle",
    "switch_tab": "Switch tab",
    "view_diagnosis": "View diagnostics",
    "browse_screener": "Browse stock screener",
    "watchlist_metric": "View watchlist indicators",
    "add_indicator": "Add Technical Indicator",
    "add_to_chart": "Add to chart",
    "view_social": "Browse community",
    "other": "Other",
    # info types
    "fundamental": "Fundamentals",
    "technical": "Technical analysis",
    "news": "新闻",
    "options": "期权",
    "social": "Social",
    "price": "Quote lookup",
    # assessment / preference
    "beginner": "Beginner",
    "intermediate": "Advanced",
    "advanced": "Proficient",
    "professional": "Professional",
    "student": "Student",
    "early_career": "Early-Career",
    "mid_career": "Mid-Career",
    "established": "Career stability",
    "retiree": "Retired",
    "active": "Active",
    "learning_active": "Actively learning",
    "passive_learning": "Passive Awareness",
    "dormant": "Inactive",
    # depth / direction
    "deep": "Depth",
    "shallow": "Shallowness",
    "evaluation": "Assessment",
    "up": "Add direction",
    "down": "Trim direction",
    # position_events emits the action UPPER-CASE. Listed explicitly rather than
    # lower-cased on lookup: a case-insensitive fallback would also translate a real
    # ticker that happens to spell a code ("ADD"), and silently rewriting a symbol
    # is far worse than leaving a code in English.
    "OPEN": "Start a new position",
    "ADD": "Add",
    "REDUCE": "Trim",
    "CLOSE": "Exit",
    "REENTER": "Start a new position",
    "yes": "Yes",
    "no": "No",
    "aligned": "Consistent",
    "divergent": "Inconsistent",
}


def _enum(value):
    """Human reading for an internal code, or the value unchanged.

    Values that are not codes (numbers, tickers, dates, free text) pass through, so
    this is safe to apply to every value position. Applied unconditionally rather
    than only under a known key: the key is often a nested path
    (`attention[].max_intent_stage`) whose leaf would otherwise miss the table, and
    every entry here is an unambiguous code.
    """
    if not isinstance(value, str):
        return value
    return ENUM_VALUES.get(value.strip(), value)


def _render_json_fragment(text):
    """Render a JSON-ish fragment as readable text, or return None.

    The profilers are asked to cite fields, and some copies do it with JSON quoting
    intact, so the reader is shown

        "price_check_share": 0.804
        "research_style"
        "event_mix": [{"event_type": "view_quote", "count": 112}]

    These are not citations with a missing explanation — they are raw data syntax at
    the presentation boundary, the same class of defect as the Python-repr leak that
    shipped twice from two different modules. Fixed where the reader is, not only at
    the producer, so a stored profile and a future producer are both covered.
    """
    s = text.strip()
    if not (s.startswith('"') or s.startswith("{") or s.startswith("[")):
        return None

    # `["a", "b"]` -> a、b  /  `["a"]` -> a
    if s.startswith("["):
        try:
            val = json.loads(s)
        except (ValueError, TypeError):
            return None
        if not isinstance(val, list):
            return None
        return "、".join(_json_value(None, v) for v in val) or None

    # `{"a": 1, "b": 2}` -> a=1、b=2
    if s.startswith("{"):
        try:
            val = json.loads(s)
        except (ValueError, TypeError):
            return None
        if not isinstance(val, dict):
            return None
        return "、".join(_json_pair(k, v) for k, v in val.items()) or None

    # `"key": value` — possibly SEVERAL pairs on one line, with no braces, because
    # the model copied an excerpt: `"event_type": "read_news", "count": 63`.
    # Wrapping in braces and parsing handles one pair and many with the same code.
    if s.startswith('"'):
        for candidate in ("{" + s + "}",):
            try:
                val = json.loads(candidate)
            except (ValueError, TypeError):
                break
            if isinstance(val, dict) and val:
                return "、".join(_json_pair(k, v) for k, v in val.items())
        # `"key"` alone
        m = _JSON_KEY.match(s)
        if m:
            label = _label_or_none(m.group(1))
            return label or m.group(1)

        # Malformed excerpt: the model dropped a colon (`"view_company", "count": 3`)
        # or the value was truncated mid-JSON. Translate whichever keys are present
        # rather than showing the reader the raw quoting.
        loose = _LOOSE_PAIR.sub(
            lambda mm: (_label_or_none(mm.group(1)) or mm.group(1)) + " ", s)
        loose = _LOOSE_BARE.sub(lambda mm: _label_or_none(mm.group(1)) or mm.group(1),
                                loose)
        loose = _LOOSE_SEP.sub("、", loose).strip('" ,')
        if loose and loose != s:
            return loose
    return None


def _label_or_none(key):
    _, entry = _lookup(key)
    return entry[0] if entry else None


def _pct_field(key):
    return (key.endswith(("_share", "_rate", "_ratio", "_pct"))
            or key in _PCT_KEYS)


def _json_value(key, val):
    """A value inside a JSON fragment, translated when it is an enum."""
    if isinstance(val, dict):
        return "、".join(_json_pair(k, v) for k, v in val.items())
    if isinstance(val, list):
        return "、".join(_json_value(key, v) for v in val)
    return str(_enum(val))


def _json_pair(key, val):
    """`price_check_share`, 0.804 -> `查价占比 80.4%`."""
    label = _label_or_none(key) or key
    if isinstance(val, float) and _pct_field(key):
        return f"{label} {val * 100:.1f}%"
    if isinstance(val, (dict, list)):
        return f"{label} {_json_value(key, val)}"
    return f"{label} {_enum(val)}"


# `"key": value` — the key is quoted. Multi-pair lines are handled by wrapping the
# whole fragment in braces and parsing it, so this is only the fallback probe.
_JSON_PAIR = re.compile(r'^"([A-Za-z_][A-Za-z0-9_.]*)"\s*:\s*(.+)$')
_JSON_KEY = re.compile(r'^"([A-Za-z_][A-Za-z0-9_.]*)"$')
# Malformed-excerpt repair: `"key": value` and bare `"key"` tokens anywhere.
_LOOSE_PAIR = re.compile(r'"([A-Za-z_][A-Za-z0-9_.]*)"\s*:\s*')
_LOOSE_BARE = re.compile(r'"([A-Za-z_][A-Za-z0-9_.]*)"')
_LOOSE_SEP = re.compile(r',\s*')
# A scalar after `field:` — a number, a quoted token, a boolean, or a percentage.
_SCALAR = re.compile(r'^[-+]?\d+(?:\.\d+)?\s*%?$|^(?:true|false|null)$|^"[^"]*"$|^×\d+$')

# Fields whose numeric value is a share, so it should be shown as a percentage.
_PCT_KEYS = {"price_check_share", "share_on_losing", "chinese_share",
             "curve_coverage", "sector_map_coverage", "top1_share",
             "top_sector_share", "top_ticker_share", "top_theme_share",
             "attention_hhi", "avg_top1_share"}


def explain(line):
    """One evidence line -> what it is, what it says, what it establishes.

    Returns `{raw, plain, meaning, known, source, field, as_of, value}`. `plain` is
    always readable; `meaning` carries the field's caveat where it has one.
    """
    raw = line if isinstance(line, str) else str(line)
    text = raw.strip()
    out = {"raw": raw, "plain": "", "meaning": "", "known": False,
           "source": None, "field": None, "as_of": None, "value": None}
    if not text:
        return out

    src, rest = _split_source(text)
    out["source"] = src
    src_zh = SOURCES.get(src, (None, None))[0] if src else None

    # A JSON fragment is rendered before anything else: it is raw data syntax, and
    # the field lookup below would only have extracted its leading key anyway.
    rendered = _render_json_fragment(text)
    if rendered is not None:
        out["plain"] = rendered
        out["known"] = True
        return out

    # Peel the leading noise off the remainder: a bare `.` separator, then any
    # leading `[date]` (which is a real fact — the snapshot date — not noise).
    body = rest.strip()
    while True:
        body = body.lstrip(".").strip()
        m = _BRACKET.match(body)
        if not m:
            break
        if out["as_of"] is None:
            out["as_of"] = m.group(1)
        body = body[m.end():].strip()

    # Split the latin field path from whatever trailing qualifier follows it.
    # NOTE the strip keeps `:` — stripping it here silently defeated the separator
    # check below, so `single_name_equity: 1.0` rendered as a qualifier ("个股占比，1.0").
    m = _PATH.match(body)
    path, tail = (m.group(1), body[m.end():].strip(" .,，")) if m else (None, "")
    # `field=value`, and `field: value` when the remainder is a SCALAR. A colon
    # followed by prose (`活跃天数: 至少发生过...` ) is a qualifier, not a value.
    if tail[:1] in ("=", ":"):
        sep, restv = tail[0], tail[1:].strip()
        if sep == "=" or _SCALAR.match(restv):
            out["value"] = restv
            tail = ""

    key, entry = (_lookup(path) if path else (None, None))
    if entry:
        out["known"] = True
        out["field"] = key
        label, meaning, *limit = entry
        limit = limit[0] if limit else None

        head = f"{src_zh}・{label}" if src_zh else label
        if out["as_of"]:
            head += f"（{out['as_of']}）"
        pieces = []
        if out["value"] is not None:
            pieces.append(f"{head} = {_enum(out['value'])}")
        else:
            pieces.append(head)
        if tail:
            pieces.append(_phrase(tail))
        out["plain"] = "：".join(pieces[:1]) + (f"，{pieces[1]}" if len(pieces) > 1 else "")
        out["meaning"] = meaning + (f"(Note: {limit})" if limit else "")
        return out

    # No field recognised. Prose is already readable; a bare token is not, and a
    # guess would be worse than the admission.
    phrased = de_repr(_phrase(body or text))
    if _HAS_CJK.search(phrased) or len(phrased.split()) > 2:
        out["plain"] = f"{src_zh}：{phrased}" if src_zh else phrased
        return out
    out["plain"] = (f"{src_zh}：{phrased}" if src_zh
                     else f"Fields without recorded definitions: {phrased}")
    out["meaning"] = "This field has not yet been included in the metric dictionary; only the original reference is retained here for verification."
    return out


def annotate(items):
    """`[str]` -> (`[raw]`, `[{plain, meaning, known, source, field, as_of}]`).

    The raw citation is preserved in full: it is the audit trail, and a reader who
    wants to verify a claim needs the exact path.
    """
    raws, notes = [], []
    for it in (items or []):
        ex = explain(it)
        raws.append(ex["raw"])
        notes.append({k: ex[k] for k in ("plain", "meaning", "known", "source",
                                         "field", "as_of")})
    return raws, notes


def coverage(lines):
    """`{total, known, source_named, bare_tokens}` — for the fleet scan and tests.

    `bare_tokens` is the list that matters: a naked latin identifier leaves the
    reader with nothing to go on, whereas a prose line or a source-qualified one is
    already readable.
    """
    known = src_named = 0
    bare = set()
    total = 0
    for it in (lines or []):
        total += 1
        ex = explain(it)
        if ex["known"]:
            known += 1
        elif ex["source"]:
            src_named += 1
        if not ex["known"] and not ex["source"] and not _HAS_CJK.search(ex["raw"]):
            bare.add(ex["raw"])
    return {"total": total, "known": known, "source_named": src_named,
            "bare_tokens": sorted(bare)}
