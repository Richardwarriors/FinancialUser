"""Per-dimension evidence projection: what each agent actually gets to see.

Why this exists
---------------
`POLICY` decides WHICH blocks a dimension may read. This module decides WHICH
KEYS inside those blocks, because the volume is concentrated in a handful of
sub-objects. Measured on a core user: one dimension received **101 KB (~25k
tokens)** of evidence, of which `behavior_intent` alone was 13.9 KB and its
`attention` list 3.45 KB.

The earlier `lite` depth only truncated list LENGTH, which saved 16% — the bulk
is scalars and nested dicts, not long lists. Trimming by length therefore cost
information without buying tokens back.

Design rules
------------
* **Whitelist, not blacklist.** A key must be named to be sent, so a new
  field is opt-in; the alternative (send everything, drop a few) silently grows
  the prompt every time the evidence layer gains a field.
* **Blocks not mentioned are kept whole.** Small blocks (≤1.4 KB) are not worth
  the risk of starving a dimension.
* **`"*"` means keep all.** For blocks where the dimension needs everything.
* **Never silent.** `project()` reports which keys/elements it removed, so a
  dimension losing evidence is visible in the plan rather than discovered later
  as a weaker narrative.
* **Element trimming is explicit.** For lists of dicts the kept element keys are
  named too, because a single list can dominate the payload.
"""
import copy
import json

# ---- small blocks kept whole (measured ≤1.4 KB each) -------------------------
# trade / position / asset / stated / conversation / behavior_trade_match /
# trading_digest / position_dynamics_digest / attention_capital_digest are all
# small enough that trimming them is not worth the risk.

_ATTENTION_FULL = ["ticker", "sector", "theme", "events_1d", "events_7d", "events_30d",
                   "events_total", "weighted_attention", "research_depth",
                   "active_days", "active_days_7d", "attention_velocity",
                   "attention_acceleration", "intent_stage"]

# `behavior_to_trade.examples` is 1.3 KB of illustrative trade/behaviour pairs on
# a 1.75 KB block. They demonstrate the matching logic; no dimension reasons over
# them, and the summary rates they illustrate are kept.
_BEHAVIOR_TO_TRADE_FULL = ["available", "lookback_days", "trades_total",
                           "trades_with_prior_research", "prior_research_rate",
                           "research_to_trade_latency_hours",
                           "pre_trade_attention_strength", "pre_trade_event_mix",
                           "pre_trade_intent_stage"]
# `alignment.largest_gaps` is useful when judging attention-vs-capital alignment;
# the ticker name lists restate it.
_ALIGNMENT = ["available", "state_counts", "dominant_state", "largest_gaps",
              "attention_gt_capital", "capital_gt_attention"]
_ALIGNMENT_MIN = ["available", "state_counts", "dominant_state"]

# Per-dimension spec. Keys named here are KEPT; anything else in that block is
# dropped. Lists of dicts may carry an element whitelist under `elem`.
DIM_EVIDENCE = {
    # risk reads only small blocks plus the position-dynamics digest; its view is
    # already ~5 KB and it is the most safety-critical dimension, so it keeps all.
    "risk": {},

    "logic": {
        # the core reasoning dimension: it keeps the full intent surface
        "behavior": {"keep": ["available", "event_count", "active_days", "date_range",
                              "top_viewed_tickers", "tickers_viewed", "etf_views",
                              "stock_views", "option_views", "timeframe_switch_count",
                              "short_timeframe_share"]},
        "behavior_intent": {
            # `targets` is a derived restatement of attention+themes (2.5 KB) and
            # `news_ticker_alignment`/`macro_topics` only describe coverage of the
            # entity extractor, not the user, so neither feeds a judgement.
            "keep": ["available", "engine", "weights_source", "totals",
                     "research_style", "research_depth", "research_breadth",
                     "exploration_vs_exploitation", "intent_funnel",
                     "tickers_by_intent_stage", "themes", "sessions_top",
                     "attention", "behavior_to_trade", "event_mix",
                     "conviction_tickers", "attention_conversion"],
            "elem": {
                "attention": _ATTENTION_FULL,
                "themes": ["theme", "attention_count", "attention_share", "trend"],
                "sessions_top": ["start", "primary_ticker", "themes", "events",
                                 "minutes", "max_intent_stage", "info_types"],
                "event_mix": ["event_type", "count", "share", "intent_stage"],
            },
            "nested": {"behavior_to_trade": {"keep": _BEHAVIOR_TO_TRADE_FULL}},
        },
        "attention_capital_digest": {"nested": {"alignment": {"keep": _ALIGNMENT}}},
    },

    "preference": {
        # style + horizon: needs what was researched and how long it was held,
        # not the trade-execution detail already covered by trading_digest.
        "behavior": {"keep": ["available", "event_count", "active_days",
                              "top_viewed_tickers", "etf_views", "stock_views",
                              "option_views", "timeframe_switch_count",
                              "short_timeframe_share"]},
        "behavior_intent": {
            "keep": ["available", "totals", "research_depth", "research_breadth",
                     "research_style", "exploration_vs_exploitation",
                     "intent_funnel", "tickers_by_intent_stage", "themes",
                     "sessions_top", "attention", "conviction_tickers"],
            "elem": {
                # horizon is read from timeframe_switches, which lives inside
                # research_depth; the rest of the per-ticker state is noise here
                "attention": ["ticker", "sector", "theme", "events_7d", "events_30d",
                              "research_depth", "active_days", "intent_stage"],
                "themes": ["theme", "attention_share", "trend"],
                "sessions_top": ["start", "primary_ticker", "themes", "events",
                                 "minutes", "max_intent_stage"],
                "event_mix": ["event_type", "share", "intent_stage"],
            },
        },
    },

    "emotion": {
        # behavioural signal only: does attention concentrate on losers, does the
        # user act fast after news. Never a full research profile.
        "behavior": {"keep": ["available", "event_count", "active_days",
                              "short_timeframe_share", "option_views"]},
        "behavior_intent": {
            "keep": ["available", "totals", "event_mix", "info_type_mix",
                     "behavior_to_trade", "attention", "intent_funnel"],
            "elem": {
                "attention": ["ticker", "theme", "events_7d", "events_30d",
                              "intent_stage", "attention_velocity"],
                "event_mix": ["event_type", "count", "share", "intent_stage"],
                "info_type_mix": ["info_type", "count", "share"],
            },
            "nested": {"behavior_to_trade": {"keep": _BEHAVIOR_TO_TRADE_FULL}},
        },
        # emotion is decided by loss-conditioned attention, not by the ticker-gap
        # table, so it gets the reduced alignment view
        "attention_capital_digest": {"nested": {"alignment": {"keep": _ALIGNMENT_MIN}}},
    },

    "assessment": {
        # expertise + lifecycle: breadth of research and persistence of interest.
        "behavior": {"keep": ["available", "event_count", "active_days",
                              "tickers_viewed", "etf_views", "stock_views",
                              "option_views", "top_viewed_tickers"]},
        "behavior_intent": {
            "keep": ["available", "totals", "research_depth", "research_breadth",
                     "exploration_vs_exploitation", "themes", "attention",
                     "conviction_tickers", "research_style"],
            "elem": {
                "attention": ["ticker", "sector", "theme", "events_30d",
                              "active_days", "intent_stage"],
                "themes": ["theme", "attention_share"],
                "event_mix": ["event_type", "share"],
            },
        },
        "attention_capital_digest": {"nested": {"alignment": {"keep": _ALIGNMENT_MIN}}},
    },
}


def _trim_list(items, allowed, path, removed):
    trimmed = []
    for item in items:
        if isinstance(item, dict):
            dropped = [ik for ik in item if ik not in allowed]
            if dropped:
                removed.setdefault("elem_keys", []).extend(
                    f"{path}.[].{ik}" for ik in dropped)
            trimmed.append({ik: iv for ik, iv in item.items() if ik in allowed})
        else:
            trimmed.append(item)
    return trimmed


def _trim_block(block, spec, removed, prefix=""):
    """Apply one block spec.

    `keep`   names top-level keys to retain (others are dropped)
    `elem`   names element keys to retain inside lists of dicts
    `nested` applies a sub-spec to a dict-valued key, so bulky sub-objects (e.g.
             behaviour_to_trade.examples, 1.3 KB of illustrative samples) can be
             trimmed without dropping the whole block
    """
    if not isinstance(block, dict):
        return block
    keep = spec.get("keep")
    elem = spec.get("elem") or {}
    nested = spec.get("nested") or {}
    out = {}
    for k, v in block.items():
        if keep is not None and k not in keep:
            removed.setdefault("keys", []).append(f"{prefix}{k}")
            continue
        if k in nested and isinstance(v, dict):
            out[k] = _trim_block(v, nested[k], removed, prefix=f"{prefix}{k}.")
        elif k in elem and isinstance(v, list):
            out[k] = _trim_list(v, set(elem[k]), f"{prefix}{k}", removed)
        else:
            out[k] = v
    return out


def project(evidence, dim=None):
    """Return the evidence view for `dim` (or the whole package if dim is None).

    Also returns what was trimmed, so a dimension losing evidence is visible
    instead of silently producing a weaker narrative.
    """
    ev = evidence or {}
    if not dim or dim not in DIM_EVIDENCE:
        return copy.deepcopy(ev), {"dimension": dim, "trimmed": False}
    spec = DIM_EVIDENCE[dim]
    removed = {}
    out = {}
    for k, v in ev.items():
        if k in spec:
            out[k] = _trim_block(v, spec[k], removed)
        else:
            out[k] = copy.deepcopy(v)
    before = len(json.dumps(ev, ensure_ascii=False, default=str))
    after = len(json.dumps(out, ensure_ascii=False, default=str))
    return out, {
        "dimension": dim,
        "trimmed": bool(removed),
        "dropped_keys": sorted(set(removed.get("keys", []))),
        "dropped_elem_keys": sorted(set(removed.get("elem_keys", []))),
        "bytes_before": before,
        "bytes_after": after,
        "reduction": round(1 - after / before, 3) if before else 0.0,
    }
