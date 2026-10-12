"""BM433 Behavior Feature Engine.

Turns the raw in-app behavior stream into structured, auditable intent
evidence. Deterministic: no LLM, every number reproducible from the snapshot.

Pipeline (mirrors the agreed BM433 design):

    Raw BM433
      -> Event Normalisation        (event_type / entity / info_type / intent)
      -> Sessionisation             (30-min gap, focus-ticker carry-forward)
      -> User-Ticker Temporal State (attention / research depth / recency /
                                     persistence / velocity / acceleration)
      -> Theme Clustering
      -> Behaviour -> Trade Alignment (conversion, latency, pre-trade mix)
      -> Derived Signals            (research style, depth, breadth,
                                     exploration vs exploitation)

Two rules are enforced throughout:

  * Nothing is inferred that the events do not support. A news headline gives
    "negative information exposure", never "user is bearish".
  * Importance weights are NOT hand-picked: ``calibrate_weights`` estimates
    P(trade within N days | event_type) from BM433 x BM452 and the engine
    loads that artifact. A documented prior is used only where the sample is
    too thin, and the output always states which one was used.
"""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .taxonomy import (extract_entities, parse_ts, strip_crypto_suffix, ticker_info)

# ---------- taxonomy ----------

INTENT_STAGES = ["awareness", "attention", "exploration", "research",
                 "evaluation", "conviction"]
_STAGE_RANK = {s: i for i, s in enumerate(INTENT_STAGES)}

# event_type -> (info_type, intent_stage, prior_lift)
# prior_lift is the *relative* importance of the event, on the same scale as the
# calibrated lift over base rate (1.0 = no better than the average event).
# It is used only when calibration cannot support an estimate.
EVENT_TYPES = {
    "read_news":        ("news",        "awareness",   0.80),
    "view_quote":       ("price",       "attention",   0.60),
    "view_social":      ("social",      "awareness",   0.40),
    "browse_screener":  ("sector",      "exploration", 1.30),
    "switch_tab":       ("context",     "exploration", 0.30),
    "view_orderbook":   ("technical",   "exploration", 1.40),
    "switch_timeframe": ("technical",   "research",    1.80),
    "add_to_chart":     ("technical",   "research",    2.00),
    "add_indicator":    ("technical",   "research",    2.00),
    "view_analysis":    ("technical",   "research",    1.50),
    "view_company":     ("fundamental", "research",    1.60),
    "view_news_page":   ("news",        "research",    1.40),
    "view_events":      ("news",        "research",    1.20),
    "view_earnings":    ("earnings",    "research",    2.20),
    "watchlist_metric": ("fundamental", "evaluation",  1.30),
    "view_diagnosis":   ("fundamental", "evaluation",  2.50),
    "view_options":     ("options",     "evaluation",  2.50),
    "other":            ("other",       "awareness",   0.20),
}

# Detail page -> event_type
_DETAIL_PAGE = {
    "price analysis": "view_analysis",
    "options chain": "view_options",
    "news & announcements": "view_news_page",
    "company info": "view_company",
    "order book": "view_orderbook",
    "financial diagnosis": "view_diagnosis",
    "events": "view_events",
    "earnings snapshot": "view_earnings",
}

# Timeframe -> horizon bucket
_TF_BUCKET = {
    "1min": "intraday", "5min": "intraday", "15min": "intraday", "1minute": "intraday",
    "1h": "intraday", "5m": "intraday", "15m": "intraday", "30m": "intraday",
    "60m": "intraday", "allday": "intraday", "pre": "intraday", "post": "intraday",
    "day": "daily", "1d": "daily",
    "week": "swing", "5day": "swing",
    "month": "position", "1month": "position", "3month": "position",
    "6month": "position", "quarter": "position", "all": "position",
    "1year": "long", "year": "long", "ytd": "long", "max": "long",
}

_SHORT_TF = {"intraday", "daily", "swing"}

# Watchlist column names reveal the metrics the user actually decides on.
_METRIC_KIND = {
    "fundamental": ["Return on net assets", "Net profit", "Return on total assets", "Price-to-earnings ratio", "Price-to-book ratio",
                    "Earnings per share", "Gross margin", "Operating revenue", "Revenue", "roe", "eps"],
    "technical":   ["Turnover", "Trading volume", "vwap", "Moving average", "macd", "rsi", "Amplitude"],
    "income":      ["Dividend Yield", "Dividend", "Dividend payout"],
    "price":       ["Price Change", "Market open", "Highest", "Lowest", "Trading value", "Price", "52 weeks"],
}

_EV_QUOTE = re.compile(
    r"^View\s+(?:(?P<cls>stock|etf|crypto|option|bonds?|economy|concept|index|futures|forex)\s+)?"
    r"(?P<tk>[A-Za-z0-9.\-]+)\s+quotes(?P<stars>\**)$")
_EV_DETAIL = re.compile(r"^(?P<tk>[A-Za-z0-9.\-]+)\s+view\s+(?P<page>.+?)(?P<stars>\**)$")
_EV_TF = re.compile(r"^Switch to (?P<tf>.+?)\s+timeframe(?P<stars>\**)$")
_EV_TAB = re.compile(r"^Switch to (?P<tab>.+?)\s+tab(?P<stars>\**)$")
_EV_NEWS = re.compile(r"^Read news:\s*(?P<title>.*?)(?P<stars>\**)$")
_EV_ADD_CHART = re.compile(r"^Add (?P<name>.+?) to chart(?P<stars>\**)$")
_EV_PATTERN = re.compile(r"^Add pattern:\s*(?P<p>\w+)(?P<stars>\**)$")
_EV_WATCHLIST = re.compile(r"^(?P<op>Add|Remove) header:\s*(?P<h>.+?)\s+(?:to|from) watchlist(?P<stars>\**)$")
_EV_SORT = re.compile(r"^Sort watchlist by:\s*(?P<k>.+?)(?P<stars>\**)$")
_EV_SCREENER = re.compile(r"^Browse screener category:\s*(?P<c>.+?)(?P<stars>\**)$")
_EV_COMMUNITY = re.compile(r"^个股社区(?P<stars>\**)$")
_EV_CONDSCREEN = re.compile(r"^条件选股(?P<stars>\**)$")
_EV_OPTION_SYM = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d+(?:\.\d+)?)$")

# Non-ticker "view X quotes" pseudo-entities we must not treat as tradables.
_NON_TICKER = {"concept", "economy", "index", "BTCUSDT"}

SESSION_GAP_MINUTES = 30
CONVICTION_LOOKBACK_DAYS = 7
WEIGHTS_PATH = Path("data/bm433_weights.json")

# ---------- helpers ----------

def _ts(s):
    """Thin alias for the shared parser (offsets, date-only, compact forms)."""
    return parse_ts(s)


def _stars(m):
    """`View stock NVDA quotes***` -> 3 extra repeats already compressed by the
    collector. We keep them as explicit repetition evidence."""
    return len(m.group("stars") or "")


def _option_underlying(sym):
    m = _EV_OPTION_SYM.match(sym or "")
    return m.group(1) if m else None


def _normalize_ticker(raw):
    """Uppercase, unwrap option symbols, strip crypto quote suffixes."""
    up = (raw or "").upper()
    und = _option_underlying(up)
    if und:
        return strip_crypto_suffix(und)
    return strip_crypto_suffix(up)


def _trade_ts(t):
    """Trade time: the real execution timestamp when available.

    Prefers `executed_at` (recovered from the raw payload) because the
    `trade_date` column is the label **extraction** date, not the trade date —
    it disagrees with reality on every row observed. The midday fallback is a
    last resort only.
    """
    ts = _ts(t.get("executed_at"))
    if ts:
        return ts
    ts = _ts(t.get("timestamp_et"))
    if ts:
        return ts
    d = t.get("trade_date")
    if not d:
        return None
    d = str(d).strip()
    return _ts(d if "T" in d else f"{d}T12:00")


def _metric_kind(name):
    low = (name or "").lower()
    for kind, kws in _METRIC_KIND.items():
        if any(k.lower() in low for k in kws):
            return kind
    return "other"


# ---------- 1. event normalisation ----------

def normalize_event(action_text):
    """Raw BM433 string -> canonical behavioral event (or None if unparseable)."""
    if not action_text:
        return None
    a = str(action_text).strip()

    m = _EV_NEWS.match(a)
    if m:
        title = m.group("title").strip().lstrip(": ").strip()
        ent = extract_entities(title)
        return {"event_type": "read_news", "ticker": ent["tickers"][0] if ent["tickers"] else None,
                "tickers": ent["tickers"], "themes": ent["themes"], "sectors": ent["sectors"],
                "macro": ent["macro"], "detail": title[:120],
                "info_type": "news", "intent_stage": "awareness", "repeat": _stars(m),
                "asset_class": "news", "timeframe": None, "metric_kind": None}

    m = _EV_QUOTE.match(a)
    if m:
        cls = (m.group("cls") or "").lower() or None
        raw_tk = m.group("tk")
        tk, asset = None, cls or "stock"
        up = raw_tk.upper()
        if cls in ("concept", "economy", "index") or up in _NON_TICKER:
            asset = cls or "index"            # macro bucket, no tradable entity
        elif cls == "option":
            tk = _option_underlying(up)
            asset = "option"
        elif cls == "crypto" or (re.fullmatch(r"[A-Z0-9]*(USDT|USDC|BUSD)", up) and len(up) > 4):
            tk, asset = strip_crypto_suffix(up), "crypto"
        else:
            tk = _normalize_ticker(up)
            asset = "etf" if (cls == "etf" or (tk in _ETF_HINT)) else (cls or "stock")
        info = ticker_info(tk) if tk else {"sector": "Other", "theme": "Other"}
        return {"event_type": "view_quote", "ticker": tk, "tickers": [tk] if tk else [],
                "themes": [info["theme"]] if info["theme"] != "Other" else [], "sectors": [],
                "macro": [], "detail": raw_tk,
                "info_type": "options" if asset == "option" else "price",
                "intent_stage": "attention", "repeat": _stars(m),
                "asset_class": asset, "timeframe": None, "metric_kind": None}

    m = _EV_TF.match(a)
    if m:
        tf = m.group("tf").strip()
        bucket = _TF_BUCKET.get(tf.lower(), "unknown")
        return {"event_type": "switch_timeframe", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": tf,
                "info_type": "technical", "intent_stage": "research", "repeat": _stars(m),
                "asset_class": "context", "timeframe": bucket, "metric_kind": None}

    m = _EV_TAB.match(a)
    if m:
        return {"event_type": "switch_tab", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": m.group("tab").strip(),
                "info_type": "context", "intent_stage": "exploration", "repeat": _stars(m),
                "asset_class": "context", "timeframe": None, "metric_kind": None}

    m = _EV_DETAIL.match(a)
    if m:
        page = m.group("page").strip().lower()
        et = _DETAIL_PAGE.get(page)
        if et:
            tk = _normalize_ticker(m.group("tk"))
            info = ticker_info(tk)
            meta = EVENT_TYPES[et]
            return {"event_type": et, "ticker": tk, "tickers": [tk] if tk else [],
                    "themes": [info["theme"]] if info["theme"] != "Other" else [],
                    "sectors": [info["sector"]] if info["sector"] != "Other" else [],
                    "macro": [], "detail": page,
                    "info_type": meta[0], "intent_stage": meta[1], "repeat": _stars(m),
                    "asset_class": "stock", "timeframe": None, "metric_kind": None}

    m = _EV_ADD_CHART.match(a)
    if m:
        return {"event_type": "add_to_chart", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": m.group("name").strip()[:40],
                "info_type": "technical", "intent_stage": "research", "repeat": _stars(m),
                "asset_class": "context", "timeframe": None, "metric_kind": None}

    m = _EV_PATTERN.match(a)
    if m:
        return {"event_type": "add_indicator", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": m.group("p"),
                "info_type": "technical", "intent_stage": "research", "repeat": _stars(m),
                "asset_class": "context", "timeframe": None, "metric_kind": None}

    m = _EV_WATCHLIST.match(a)
    if m:
        h = m.group("h").strip()
        return {"event_type": "watchlist_metric", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": h,
                "info_type": _metric_kind(h), "intent_stage": "evaluation", "repeat": _stars(m),
                "asset_class": "context", "timeframe": None, "metric_kind": _metric_kind(h)}

    m = _EV_SORT.match(a)
    if m:
        k = m.group("k").strip()
        return {"event_type": "watchlist_metric", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": k,
                "info_type": _metric_kind(k), "intent_stage": "evaluation", "repeat": _stars(m),
                "asset_class": "context", "timeframe": None, "metric_kind": _metric_kind(k)}

    m = _EV_SCREENER.match(a)
    if m:
        return {"event_type": "browse_screener", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": m.group("c").strip()[:40],
                "info_type": "technical", "intent_stage": "exploration", "repeat": _stars(m),
                "asset_class": "context", "timeframe": None, "metric_kind": None}

    if _EV_COMMUNITY.match(a):
        return {"event_type": "view_social", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": "community",
                "info_type": "social", "intent_stage": "awareness", "repeat": 0,
                "asset_class": "context", "timeframe": None, "metric_kind": None}

    if _EV_CONDSCREEN.match(a):
        return {"event_type": "browse_screener", "ticker": None, "tickers": [], "themes": [],
                "sectors": [], "macro": [], "detail": "conditional_screen",
                "info_type": "technical", "intent_stage": "exploration", "repeat": 0,
                "asset_class": "context", "timeframe": None, "metric_kind": None}

    return {"event_type": "other", "ticker": None, "tickers": [], "themes": [], "sectors": [],
            "macro": [], "detail": a[:60], "info_type": "other", "intent_stage": "awareness",
            "repeat": 0, "asset_class": "context", "timeframe": None, "metric_kind": None}


_ETF_HINT = {"QQQ", "SPY", "IWM", "DRAM", "SOXL", "SOXS", "TQQQ", "SQQQ", "NVDL",
             "USDY", "SGOV", "VOO", "VTI", "DIA", "LITX", "GOOX"}


def normalize_stream(behaviors):
    """Snapshot behaviors -> ordered canonical events with timestamps."""
    out = []
    for b in behaviors:
        ev = normalize_event(b.get("action_text"))
        if not ev:
            continue
        ev["ts"] = _ts(b.get("timestamp_et")) or _ts(b.get("event_date"))
        ev["date"] = b.get("event_date") or (ev["ts"].date().isoformat() if ev["ts"] else None)
        if ev["ts"] is None:
            continue
        out.append(ev)
    out.sort(key=lambda e: e["ts"])
    return out


# ---------- 2. sessionisation ----------

def build_sessions(events, gap_minutes=SESSION_GAP_MINUTES):
    """Group events into research sessions; carry the active ticker forward so a
    bare timeframe switch or chart tool is attributed to the ticker under
    study (that is what makes 'research depth per ticker' meaningful)."""
    sessions = []
    cur = None
    for e in events:
        if cur is None or (e["ts"] - cur["last_ts"]).total_seconds() > gap_minutes * 60:
            cur = {"start": e["ts"], "last_ts": e["ts"], "events": []}
            sessions.append(cur)
        cur["events"].append(e)
        cur["last_ts"] = e["ts"]

    for s in sessions:
        focus = None
        for e in s["events"]:
            if e.get("ticker"):
                focus = e["ticker"]
            e["focus_ticker"] = e.get("ticker") or focus
        tickers = [e["focus_ticker"] for e in s["events"] if e.get("focus_ticker")]
        cnt = Counter(tickers)
        s["primary_ticker"] = cnt.most_common(1)[0][0] if cnt else None
        s["related_tickers"] = [t for t, _ in cnt.most_common(6) if t != s["primary_ticker"]]
        s["minutes"] = round((s["last_ts"] - s["start"]).total_seconds() / 60, 1)
        s["n_events"] = len(s["events"])
        stages = [e["intent_stage"] for e in s["events"]]
        s["max_stage"] = max(stages, key=lambda x: _STAGE_RANK[x]) if stages else "awareness"
        themes = Counter()
        for e in s["events"]:
            for t in e.get("themes") or []:
                themes[t] += 1
        s["themes"] = [t for t, _ in themes.most_common(3)]
        s["info_types"] = sorted({e["info_type"] for e in s["events"] if e["info_type"] != "context"})
        s["event_types"] = sorted({e["event_type"] for e in s["events"]})
    return sessions


# ---------- 3. weights (empirical, from BM433 x BM452) ----------

PRIOR_WEIGHTS = {k: v[2] for k, v in EVENT_TYPES.items()}   # lift scale, 1.0 = average


def load_weights(path=WEIGHTS_PATH):
    """Calibrated lift-over-base-rate weights if available, else the prior."""
    p = Path(path)
    if p.exists():
        try:
            d = json.loads(p.read_text())
            tbl = d.get("weights") or {}
            if tbl:
                return {"source": f"empirical:{d.get('computed_at', 'unknown')}",
                        "scale": "lift_vs_base_rate",
                        "base_rate": d.get("base_rate"),
                        "table": {**PRIOR_WEIGHTS, **tbl}, "samples": d.get("samples", {})}
        except Exception:
            pass
    return {"source": "prior", "scale": "lift_vs_base_rate",
            "table": dict(PRIOR_WEIGHTS), "samples": {}}


def calibrate_weights(db_path, horizon_days=3, min_exposure=30, out_path=WEIGHTS_PATH,
                      min_users=5):
    """Estimate P(trade on the same ticker within `horizon_days` | event_type).

    Exposure = (user, ticker, day) pairs where that event_type occurred.
    Positive  = a BM452 trade on the same ticker within the horizon.

    Raw rates are noisy for rare event types, so they are shrunk toward the
    global base rate (Bayesian smoothing, k = min_exposure). A weight is only
    adopted when it is backed by BOTH enough exposures and enough distinct
    users — otherwise a single heavy trader would set the weights for
    everybody, and the documented prior is kept instead.
    """
    import sqlite3
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    users = [r[0] for r in c.execute("select distinct user_id from events")]

    exposures = Counter()
    positives = Counter()
    by_user = defaultdict(set)
    users_used = 0
    for uid in users:
        beh = [dict(r) for r in c.execute(
            "select event_date,timestamp_et,action_text from events where user_id=? order by timestamp_et",
            (uid,))]
        trd = [dict(r) for r in c.execute(
            "select trade_date,executed_at,ticker,side from bm452_trades where user_id=?", (uid,))]
        if not beh or not trd:
            continue
        trades = []
        for t in trd:
            tk = _normalize_ticker(t.get("ticker"))
            dt = _trade_ts(t)
            if tk and dt:
                trades.append((tk, dt))
        if not trades:
            continue
        users_used += 1

        # Sessionise first: a timeframe switch or chart tool carries no ticker
        # of its own, and without the focus-ticker carry-forward those 24k+
        # context events would never be counted as exposure at all.
        evs = normalize_stream(beh)
        if not evs:
            continue
        build_sessions(evs)

        # group exposures by (ticker, day): each day counts once per event type,
        # so a user who spams quote views does not dominate the estimate
        by_key = defaultdict(lambda: {"types": set(), "first_ts": None})
        for e in evs:
            tk = e.get("focus_ticker")
            d = e.get("date")
            if not tk or not d:
                continue
            entry = by_key[(tk, d)]
            entry["types"].add(e["event_type"])
            if entry["first_ts"] is None or e["ts"] < entry["first_ts"]:
                entry["first_ts"] = e["ts"]
        for (tk, _d), entry in by_key.items():
            ts = entry["first_ts"]
            hit = any(tt == tk and timedelta(0) <= (td - ts) <= timedelta(days=horizon_days)
                      for tt, td in trades)
            for et in entry["types"]:
                exposures[et] += 1
                by_user[et].add(uid)
                if hit:
                    positives[et] += 1

    total_exp = sum(exposures.values())
    total_pos = sum(positives.values())
    base = (total_pos / total_exp) if total_exp else 0.0
    k = min_exposure
    weights, samples = {}, {}
    for et in EVENT_TYPES:
        n, p = exposures.get(et, 0), positives.get(et, 0)
        nu = len(by_user.get(et, ()))
        rate = (p / n) if n else None
        lift = (rate / base) if rate is not None and base else None
        samples[et] = {"exposures": n, "positives": p, "users": nu,
                       "raw_rate": round(rate, 4) if rate is not None else None,
                       "lift_vs_base": round(lift, 2) if lift is not None else None,
                       "adopted": n >= min_exposure and nu >= min_users}
        if samples[et]["adopted"]:
            # smoothed lift: shrinkage keeps a 0/104 event type from claiming
            # "zero signal" and a 1/5 one from claiming a huge multiplier
            smoothed_rate = (p + k * base) / (n + k)
            weights[et] = round(smoothed_rate / base, 2) if base else PRIOR_WEIGHTS[et]
        else:
            weights[et] = PRIOR_WEIGHTS[et]
    artifact = {"computed_at": datetime.now().isoformat(timespec="seconds"),
                "horizon_days": horizon_days, "base_rate": round(base, 4),
                "min_exposure": min_exposure, "min_users": min_users,
                "scale": "lift_vs_base_rate",
                "note": ("weight = P(trade on same ticker within horizon | event type "
                         "seen that day) divided by the global base rate; 1.0 = average, "
                         ">1 = better than average predictor. Bayesian-shrunk."),
                "users_with_trades": users_used,
                "weights": weights, "samples": samples}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(artifact, ensure_ascii=False, indent=1))
    return artifact


# ---------- 4. per-ticker temporal state ----------

def _rate(count, days):
    return count / days if days else 0.0


def build_ticker_states(events, weights, ref, top_n=10, conviction_tickers=None):
    """User x ticker dynamic state: attention, research depth, recency,
    persistence, velocity and acceleration.

    `conviction_tickers` are tickers whose research actually resolved into a
    real trade — the only basis on which the engine will claim conviction.
    """
    conviction_tickers = conviction_tickers or set()
    by_tk = defaultdict(list)
    for e in events:
        tk = e.get("focus_ticker") or e.get("ticker")
        if tk:
            by_tk[tk].append(e)

    states = []
    for tk, evs in by_tk.items():
        evs.sort(key=lambda e: e["ts"])
        last = evs[-1]["ts"]
        recency_h = round((ref - last).total_seconds() / 3600, 1)
        days = {e["date"] for e in evs if e["date"]}

        def in_window(d):
            return [e for e in evs if (ref - e["ts"]) <= timedelta(days=d)]

        w1, w7, w30 = in_window(1), in_window(7), in_window(30)
        depth = Counter(e["event_type"] for e in evs)
        weighted = sum((1 + e.get("repeat", 0)) * weights.get(e["event_type"], 0.05) for e in evs)

        # velocity: recent 7d rate vs the prior 21d baseline
        prior21 = [e for e in evs if timedelta(days=7) < (ref - e["ts"]) <= timedelta(days=28)]
        base_rate = _rate(len(prior21), 21)
        velocity = round(_rate(len(w7), 7) / base_rate, 2) if base_rate > 0 else (
            round(_rate(len(w7), 7), 2) if w7 else 0.0)
        # acceleration: last 3d rate vs the 30d rate
        r30 = _rate(len(w30), 30)
        accel = round(_rate(len(in_window(3)), 3) / r30, 2) if r30 > 0 else (
            round(_rate(len(in_window(3)), 3), 2) if in_window(3) else 0.0)

        stage = max((e["intent_stage"] for e in evs), key=lambda s: _STAGE_RANK[s])
        # conviction is only claimable when a real trade followed the research
        if tk in conviction_tickers:
            stage = "conviction"

        info = ticker_info(tk)
        # only non-zero depth counters are emitted: an all-zero research_depth
        # block is pure token waste in every prompt
        depth_full = {"news": depth["read_news"] + depth["view_news_page"],
                      "chart_tools": depth["add_to_chart"] + depth["add_indicator"],
                      "timeframe_switches": depth["switch_timeframe"],
                      "fundamental": depth["view_company"] + depth["view_diagnosis"],
                      "options": depth["view_options"],
                      "earnings": depth["view_earnings"],
                      "orderbook": depth["view_orderbook"]}
        states.append({
            "ticker": tk,
            "sector": info["sector"], "theme": info["theme"],
            "events_1d": len(w1), "events_7d": len(w7), "events_30d": len(w30),
            "events_total": len(evs),
            "weighted_attention": round(weighted, 3),
            "research_depth": {k: v for k, v in depth_full.items() if v},
            "last_event_hours": recency_h,
            "first_seen": evs[0]["date"], "last_seen": evs[-1]["date"],
            "active_days": len(days),
            "active_days_7d": len({e["date"] for e in w7}),
            "repeat_markers": sum(e.get("repeat", 0) for e in evs),
            "attention_velocity": velocity,
            "attention_acceleration": accel,
            "intent_stage": stage,
        })
    states.sort(key=lambda s: -s["weighted_attention"])
    return states[:top_n], len(by_tk)


# ---------- 5. behaviour -> trade alignment ----------

def align_with_trades(events, trades, lookback_days=7, weights=None, top_n=8):
    """For each real trade, summarise the research that preceded it."""
    weights = weights or PRIOR_WEIGHTS
    tl = []
    for t in trades:
        tk = _normalize_ticker(t.get("ticker"))
        ts = _trade_ts(t)
        if tk and ts:
            tl.append({"ticker": tk, "ts": ts, "side": t.get("side")})
    if not tl:
        return {"available": False}

    aligned, latencies, pre_mix, depth_counts, strengths = [], [], Counter(), Counter(), []
    for tr in sorted(tl, key=lambda x: x["ts"]):
        win = [e for e in events
               if (e.get("focus_ticker") or e.get("ticker")) == tr["ticker"]
               and timedelta(0) <= (tr["ts"] - e["ts"]) <= timedelta(days=lookback_days)]
        stages = [e["intent_stage"] for e in win]
        stage = max(stages, key=lambda s: _STAGE_RANK[s]) if stages else None
        # strength of pre-trade attention: lift-weighted event volume, not a raw count
        strength = sum((1 + e.get("repeat", 0)) * weights.get(e["event_type"], 0.2) for e in win)
        strengths.append(strength)
        if win:
            depth_counts[stage] += 1
            pre_mix.update(e["event_type"] for e in win)
            latencies.append((tr["ts"] - win[0]["ts"]).total_seconds() / 3600)
            aligned.append({"ticker": tr["ticker"], "side": tr["side"],
                            "trade_date": tr["ts"].date().isoformat(),
                            "research_events": len(win),
                            "attention_strength": round(strength, 1),
                            "max_intent_stage": stage,
                            "research_days": len({e["date"] for e in win})})
    n = len(tl)
    latencies.sort()
    strengths.sort()
    return {
        "available": True,
        "lookback_days": lookback_days,
        "trades_total": n,
        "trades_with_prior_research": len(aligned),
        "prior_research_rate": round(len(aligned) / n, 3) if n else None,
        "research_to_trade_latency_hours": {
            "median": round(latencies[len(latencies) // 2], 1) if latencies else None,
            "p25": round(latencies[len(latencies) // 4], 1) if latencies else None,
            "p75": round(latencies[len(latencies) * 3 // 4], 1) if latencies else None,
        },
        "pre_trade_attention_strength": {
            "median": round(strengths[len(strengths) // 2], 1) if strengths else None,
            "min": round(strengths[0], 1) if strengths else None,
            "max": round(strengths[-1], 1) if strengths else None,
        },
        "pre_trade_event_mix": [{"event_type": k, "count": v} for k, v in pre_mix.most_common(8)],
        "pre_trade_intent_stage": dict(depth_counts),
        "examples": aligned[:top_n],
    }


def attention_conversion(events, trades, horizon_days=3):
    """Attention -> action conversion, with negatives (viewed, no trade).

    Unit of analysis is (user, ticker, day): does attention on that ticker
    resolve into a trade within the horizon?
    """
    tl = []
    for t in trades:
        tk = _normalize_ticker(t.get("ticker"))
        dts = _trade_ts(t)
        if tk and dts:
            tl.append((tk, dts))
    if not tl or not events:
        return {"available": False}

    by_key = defaultdict(list)
    for e in events:
        tk = e.get("focus_ticker") or e.get("ticker")
        if tk and e["ts"]:
            by_key[(tk, e["date"])].append(e)

    by_theme_pos, by_theme_tot = Counter(), Counter()
    pos = tot = 0
    deep_pos = deep_tot = 0
    for (tk, d), evs in by_key.items():
        ts = min(e["ts"] for e in evs)
        tot += 1
        theme = ticker_info(tk)["theme"]
        by_theme_tot[theme] += 1
        hit = any(tt == tk and timedelta(0) <= (td - ts) <= timedelta(days=horizon_days)
                  for tt, td in tl)
        max_stage = max((e["intent_stage"] for e in evs), key=lambda s: _STAGE_RANK[s])
        if _STAGE_RANK[max_stage] >= _STAGE_RANK["research"]:
            deep_tot += 1
            if hit:
                deep_pos += 1
        if hit:
            pos += 1
            by_theme_pos[theme] += 1
    return {
        "available": True,
        "horizon_days": horizon_days,
        "attended_ticker_days": tot,
        "converted_ticker_days": pos,
        "view_to_trade_conversion": round(pos / tot, 4) if tot else None,
        "deep_research_to_trade_conversion": round(deep_pos / deep_tot, 4) if deep_tot else None,
        "deep_research_days": deep_tot,
        "by_theme": [{"theme": k, "attend": v,
                      "convert": by_theme_pos.get(k, 0),
                      "rate": round(by_theme_pos.get(k, 0) / v, 4)}
                     for k, v in by_theme_tot.most_common(8) if k != "Other"],
    }


# ---------- 6. user-level derived signals ----------

def _ticker_max_stage(events, conviction_tickers=None):
    """Max intent stage reached per ticker (context-only events excluded)."""
    conviction_tickers = conviction_tickers or set()
    out = {}
    for e in events:
        tk = e.get("focus_ticker")
        if not tk:
            continue
        cur = out.get(tk)
        if cur is None or _STAGE_RANK[e["intent_stage"]] > _STAGE_RANK[cur]:
            out[tk] = e["intent_stage"]
    for tk in conviction_tickers:
        if tk in out:
            out[tk] = "conviction"
    return out

def _research_style(events):
    counts = Counter(e["info_type"] for e in events)
    total = sum(v for k, v in counts.items() if k not in ("context", "other"))
    if not total:
        return {}
    return {
        "news_driven": round(counts["news"] / total, 3),
        "technical_driven": round(counts["technical"] / total, 3),
        "fundamental_driven": round((counts["fundamental"] + counts["earnings"]) / total, 3),
        "options_driven": round(counts["options"] / total, 3),
        "social_driven": round(counts["social"] / total, 3),
        "price_check_share": round(counts["price"] / total, 3),
    }


def _position_context(events, holdings_latest, trades, window_days=30):
    """Label each event held / recently_bought / recently_sold / not_held using
    the BM453 book and the BM452 history (design section 13)."""
    held = {(k or "").upper() for k in (holdings_latest or {})}
    buys, sells = set(), set()
    ref = events[-1]["ts"] if events else None
    for t in trades or []:
        tk = _normalize_ticker(t.get("ticker"))
        ts = _trade_ts(t)
        if not tk or not ts or ref is None:
            continue
        if (ref - ts) > timedelta(days=window_days):
            continue
        if t.get("side") == "buy":
            buys.add(tk)
        elif t.get("side") == "sell":
            sells.add(tk)
    mix = Counter()
    held_events = 0
    for e in events:
        tk = e.get("focus_ticker") or e.get("ticker")
        if not tk:
            continue
        if tk in held:
            ctx = "held"
            held_events += 1
        elif tk in sells:
            ctx = "recently_sold"
        elif tk in buys:
            ctx = "recently_bought"
        else:
            ctx = "not_held"
        mix[ctx] += 1
    return mix, held_events


# ---------- 7. top-level engine ----------

def _news_alignment(events):
    """How often reading news links back to tickers the user actually follows.

    The BM433 news corpus is English and macro-dominated (geopolitics and
    politics outweigh single-name stories), so requiring a ticker would score
    most headlines as "no signal" when they in fact carry a clear topic. Both a
    ticker link and a macro topic are therefore reported; a headline with
    neither is not counted either way.
    """
    followed = {(e.get("focus_ticker") or e.get("ticker"))
                for e in events if e.get("focus_ticker") or e.get("ticker")}
    news = [e for e in events if e["event_type"] in ("read_news", "view_news_page")]
    linked = [e for e in news if e.get("tickers")]
    aligned = [e for e in linked if any(t in followed for t in e["tickers"])]
    macro = [e for e in news if e.get("macro")]
    macro_mix = Counter(m for e in macro for m in e["macro"])
    entity_rate = (len(linked) / len(news)) if news else None
    macro_rate = (len(macro) / len(news)) if news else None
    return {
        "news_events": len(news),
        "news_with_entity": len(linked),
        "entity_extraction_rate": round(entity_rate, 3) if entity_rate is not None else None,
        "ticker_linked_to_followed": len(aligned),
        "alignment_rate": round(len(aligned) / len(linked), 3) if linked else None,
        "macro_tagged": len(macro),
        "macro_tagging_rate": round(macro_rate, 3) if macro_rate is not None else None,
        "macro_topics": [{"topic": k, "count": v} for k, v in macro_mix.most_common(8)],
        "usable_signal_rate": round(
            (len({id(e) for e in linked} | {id(e) for e in macro}) / len(news)), 3) if news else None,
        "note": ("High proportion of macroeconomic news: no corresponding individual stock instrument, but macro themes are separately labeled and must not be treated as no information due to the lack of individual stocks."), 
    }


def _build_targets(states, theme_rows, holdings, conviction, ticker_att, n_events):
    """Three explicit target levels (design section 14).

    Level 1 Attention  -- what the user is currently focused on.
    Level 2 Intent     -- what they may be preparing to act on (researched but
                          not held; attention accelerating).
    Level 3 Action     -- research that already resolved into a real trade.

    Levels 2 and 3 are claims about behavior, not predictions; level 2 is
    explicitly labelled as a candidate, never a forecast.
    """
    held = {(k or "").upper() for k in (holdings or {})}
    total = sum(ticker_att.values()) or 1

    l1 = [{"ticker": s["ticker"], "theme": s["theme"],
           "attention_share": round(ticker_att.get(s["ticker"], 0) / total, 3),
           "events_7d": s["events_7d"], "events_30d": s["events_30d"],
           "intent_stage": s["intent_stage"]} for s in states[:5]]

    l2 = []
    for s in states:
        if s["ticker"] in held or s["ticker"] in conviction:
            continue
        if _STAGE_RANK[s["intent_stage"]] < _STAGE_RANK["research"]:
            continue
        l2.append({"ticker": s["ticker"], "theme": s["theme"],
                   "intent_stage": s["intent_stage"],
                   "not_currently_held": True,
                   "attention_acceleration": s["attention_acceleration"],
                   "research_depth": s["research_depth"],
                   "active_days": s["active_days"]})
    l2 = sorted(l2, key=lambda x: -x["attention_acceleration"])[:5]

    l3 = [{"ticker": s["ticker"], "theme": s["theme"],
           "basis": "research preceded a real trade",
           "currently_held": s["ticker"] in held,
           "events_total": s["events_total"],
           "last_seen": s["last_seen"]}
          for s in states if s["ticker"] in conviction]

    return {
        "level1_attention": l1,
        "level1_top_themes": [{"theme": t["theme"], "attention_share": t["attention_share"],
                               "trend": t["trend"]} for t in theme_rows[:4]],
        "level2_intent_candidates": l2,
        "level2_note": ("researched/repeatedly checked but not in the latest BM453 "
                        "book; a candidate under consideration, not a prediction"),
        "level3_action_observed": l3,
        "level3_note": "retrospective: research on these tickers resolved into trades",
    }


def build_behavior_intent(snap, weights_path=WEIGHTS_PATH, max_tickers=8):
    """Full BM433 engine output for one snapshot."""
    behaviors = snap.get("behaviors_6m") or []
    if not behaviors:
        return {"available": False}

    events = normalize_stream(behaviors)
    if not events:
        return {"available": False}

    ref = events[-1]["ts"]
    w = load_weights(weights_path)
    weights = w["table"]

    n = len(events)
    days = {e["date"] for e in events}
    unparsed = sum(1 for e in events if e["event_type"] == "other")
    repeated = sum(e.get("repeat", 0) for e in events)

    # event / info mix
    et_counts = Counter(e["event_type"] for e in events)
    it_counts = Counter(e["info_type"] for e in events)
    stage_counts = Counter(e["intent_stage"] for e in events)

    # sessions
    sessions = build_sessions(events)
    sess_len = sorted(s["n_events"] for s in sessions)
    sess_min = sorted(s["minutes"] for s in sessions)
    deep_sessions = [s for s in sessions if _STAGE_RANK[s["max_stage"]] >= _STAGE_RANK["research"]]

    trades = snap.get("trades_6m") or []
    # conviction = a ticker whose research actually resolved into a real trade
    conviction = set()
    for t in trades:
        tk = _normalize_ticker(t.get("ticker"))
        ts = _trade_ts(t)
        if not tk or not ts:
            continue
        if any((e.get("focus_ticker") == tk and timedelta(0) <= (ts - e["ts"])
                <= timedelta(days=CONVICTION_LOOKBACK_DAYS)) for e in events):
            conviction.add(tk)

    # ticker states
    states, n_tickers = build_ticker_states(events, weights, ref, top_n=max_tickers,
                                           conviction_tickers=conviction)

    # themes
    theme_att = Counter()
    for e in events:
        for t in e.get("themes") or []:
            theme_att[t] += 1
    theme_share = {k: round(v / n, 3) for k, v in theme_att.items()}
    # recent vs prior 30d per theme -> trend
    recent_themes, prior_themes = Counter(), Counter()
    for e in events:
        age = (ref - e["ts"]).days
        for t in e.get("themes") or []:
            if age < 7:
                recent_themes[t] += 1
            elif age < 37:
                prior_themes[t] += 1
    theme_rows = []
    for k, v in theme_att.most_common(8):
        rr, pr = recent_themes.get(k, 0) / 7, prior_themes.get(k, 0) / 30
        trend = ("increasing" if rr > pr * 1.3 else "declining" if rr < pr * 0.7 else "stable")
        theme_rows.append({"theme": k, "attention_count": v, "attention_share": theme_share[k],
                           "trend": trend})

    # breadth / exploration vs exploitation
    holdings = {}
    acct = snap.get("account_daily") or []
    if acct:
        holdings = acct[-1].get("holdings") or {}
    held = {(k or "").upper() for k in holdings}
    ctx_mix, held_events = _position_context(events, holdings, snap.get("trades_6m") or [])
    ticker_att = Counter()
    for e in events:
        tk = e.get("focus_ticker") or e.get("ticker")
        if tk:
            ticker_att[tk] += 1
    hhi = (sum((v / sum(ticker_att.values())) ** 2 for v in ticker_att.values())
           if ticker_att else 0.0)

    # macrotopics
    macro = Counter()
    for e in events:
        for m in e.get("macro") or []:
            macro[m] += 1

    trades = snap.get("trades_6m") or []
    align = align_with_trades(events, trades, weights=weights)
    conv = attention_conversion(events, trades)

    # per-ticker funnel: how many tickers reached each intent stage at all
    ticker_stage = Counter(_ticker_max_stage(events, conviction).values())
    tickers_by_stage = {s: ticker_stage.get(s, 0) for s in INTENT_STAGES}

    style = _research_style(events)
    depth_ratio = len(deep_sessions) / len(sessions) if sessions else None
    targets = _build_targets(states, theme_rows, holdings, conviction, ticker_att, n)

    return {
        "available": True,
        "engine": "bm433-behavior-v1",
        "weights_source": w["source"],
        "weights_scale": "lift_vs_base_rate",
        "totals": {
            "events": n,
            "active_days": len(days),
            "date_range": [min(days), max(days)] if days else None,
            "unique_tickers": n_tickers,
            "avg_events_per_active_day": round(n / len(days), 1) if days else None,
            "repeat_marker_events": repeated,
            "unparsed_events": unparsed,
            "unparsed_share": round(unparsed / n, 4),
        },
        "event_mix": [{"event_type": k, "count": v, "share": round(v / n, 3),
                       "info_type": EVENT_TYPES[k][0], "intent_stage": EVENT_TYPES[k][1],
                       "weight": weights.get(k)} for k, v in et_counts.most_common(14)],
        "info_type_mix": [{"info_type": k, "count": v, "share": round(v / n, 3)}
                          for k, v in it_counts.most_common(10)],
        "intent_funnel": {s: stage_counts.get(s, 0) for s in INTENT_STAGES},
        "tickers_by_intent_stage": tickers_by_stage,
        "research_style": style,
        "news_ticker_alignment": _news_alignment(events),
        "research_depth": {
            "sessions": len(sessions),
            "research_grade_sessions": len(deep_sessions),
            "research_grade_session_share": round(depth_ratio, 3) if depth_ratio is not None else None,
            "median_session_events": sess_len[len(sess_len) // 2] if sess_len else None,
            "median_session_minutes": sess_min[len(sess_min) // 2] if sess_min else None,
            "max_session_events": sess_len[-1] if sess_len else None,
        },
        "research_breadth": {
            "unique_tickers": n_tickers,
            "attention_hhi": round(hhi, 4),
            "top_ticker_share": round(max(ticker_att.values()) / sum(ticker_att.values()), 3)
            if ticker_att else None,
            "top_theme_share": theme_rows[0]["attention_share"] if theme_rows else None,
        },
        "exploration_vs_exploitation": {
            "held_ticker_event_share": round(held_events / sum(ctx_mix.values()), 3)
            if sum(ctx_mix.values()) else None,
            "new_ticker_event_share": round(ctx_mix.get("not_held", 0) / sum(ctx_mix.values()), 3)
            if sum(ctx_mix.values()) else None,
            "position_context_mix": dict(ctx_mix),
        },
        "attention": states,
        "themes": theme_rows,
        "targets": targets,
        "macro_topics": [{"topic": k, "count": v} for k, v in macro.most_common(6)],
        "sessions_top": [{"start": s["start"].isoformat(timespec="minutes"),
                          "primary_ticker": s["primary_ticker"],
                          "themes": s["themes"], "events": s["n_events"],
                          "minutes": s["minutes"], "max_intent_stage": s["max_stage"],
                          "info_types": s["info_types"]}
                         for s in sorted(sessions, key=lambda x: -x["n_events"])[:4]],
        "behavior_to_trade": align,
        "attention_conversion": conv,
        "conviction_tickers": sorted(conviction),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="BM433 behavior engine utilities")
    ap.add_argument("--calibrate", action="store_true", help="recompute empirical weights")
    ap.add_argument("--db", default="data/bm.db")
    ap.add_argument("--horizon-days", type=int, default=3)
    ap.add_argument("--min-exposure", type=int, default=30)
    ap.add_argument("--min-users", type=int, default=5)
    ap.add_argument("--out", default=str(WEIGHTS_PATH))
    ap.add_argument("--dump-weights", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        art = calibrate_weights(a.db, a.horizon_days, a.min_exposure, a.out, a.min_users)
        print(json.dumps({"computed_at": art["computed_at"], "base_rate": art["base_rate"],
                          "horizon_days": art["horizon_days"], "weights": art["weights"]},
                         ensure_ascii=False, indent=1))
        print("samples:", json.dumps({k: v for k, v in art["samples"].items()},
                                     ensure_ascii=False))
    elif a.dump_weights:
        print(json.dumps(load_weights(a.out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
