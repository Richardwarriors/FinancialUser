"""Project a rolling sample down to what the predictor is allowed to see.

A sample file is an ARCHIVE: it carries the full evidence tree and every computed
block so a result can be audited or ablated later.  That makes it ~63k tokens,
and 93% of it is the profile snapshot's audit layers:

    evidence_summary   ~30k tok   raw computed metric blocks
    presentation       ~25k tok   profile_tree / kyc_tree evidence rendering
    structured          ~2k tok   the dimension conclusions
    base_profile        ~1k tok   the headline narrative

Feeding the archive to a model is both wasteful and unsafe: the two audit layers
restate the same evidence three ways (`evidence`, `evidence_plain`, `display`),
which crowds out the recent window that actually carries the next-day signal.

`build_prompt_view` keeps the conclusion layers plus the short-term evidence and
drops the audit layers, landing at ~7k tokens.  It is a projection, never a
rewrite: every field it emits is copied from the sample unchanged, so a
prediction can always be traced back to the archived sample.
"""
from __future__ import annotations

import json

from .rolling_schema import public_profile_contamination

# Measured on the 80-user cohort: the compact view sits near 7k tokens. The cap is
# a tripwire for a future field that quietly reintroduces an audit layer.
TOKEN_BUDGET = 12000
CHARS_PER_TOKEN = 4

# `observation_coverage` is label PROVENANCE: it records which boundary books were
# observed so an `unknown` label can be told apart from a real `hold`. Most of it
# is measured ON or AFTER the target day -- `after_book_observed` says the label
# exists, `after_book_as_of` is the prediction date itself, and
# `execution_coverage_basis` names a date weeks past the cutoff. A point-in-time
# predictor cannot know any of that, so only the cutoff-side facts are shown.
COVERAGE_SHOWN = ("before_book_observed", "before_book_as_of")


def _tokens(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          default=str)) // CHARS_PER_TOKEN


def build_prompt_view(sample: dict, *, budget: int = TOKEN_BUDGET) -> dict:
    """-> the model-facing view of one sample. Raises if it exceeds `budget`."""
    profile = sample.get("profile_snapshot") or {}
    recent = sample.get("recent_7d") or {}
    coverage = sample.get("observation_coverage") or {}
    base = profile.get("base_profile") or {}
    # Fail loudly on schema drift rather than emitting a view full of nulls: a
    # silently empty prompt is indistinguishable from a user with no history.
    missing = [name for name, node in (("profile_snapshot.base_profile", base),
                                       ("profile_snapshot.structured", profile.get("structured")),
                                       ("recent_7d", recent),
                                       ("market_7d", sample.get("market_7d")))
               if not node]
    if missing:
        raise ValueError(f"sample {sample.get('sample_id')} is missing {missing}; "
                         "the prompt view would be empty")
    for required in ("window_start", "window_end", "trading_days"):
        if required not in recent:
            raise ValueError(f"recent_7d has no {required!r}; schema changed")
    view = {
        "cutoff_et": sample.get("cutoff_et"),
        "prediction_date": sample.get("prediction_date"),
        "user_id": sample.get("user_id"),
        # Long horizon: the expanding profile's conclusions only.
        "profile": {
            "as_of": profile.get("cutoff_et"),
            "version": profile.get("profile_version"),
            "headline": base.get("headline"),
            "summary": base.get("summary"),
            "dimensions": base.get("dimensions"),
            "caveats": base.get("caveats"),
            "traits": profile.get("structured"),
            "data_quality": profile.get("data_quality"),
        },
        # Short horizon: passed through whole. Every field in it is signal, and at
        # ~3k tokens it is not worth the fragility of re-listing its subkeys.
        "recent_7d": recent,
        "market_7d": sample.get("market_7d"),
        "observation_coverage": {k: coverage[k] for k in COVERAGE_SHOWN
                                if k in coverage},
        "note": (
            "profile is the expanding history as of the cutoff and describes habitual "
            "behaviour. recent_7d covers the seven trading days before the cutoff. "
            "market_7d is point-in-time and contains no bar after the cutoff. "
            "A missing or null field means it was not observed, never that it was zero."
        ),
    }
    leaked = public_profile_contamination(view["profile"])
    if leaked:
        raise ValueError(
            f"sample {sample.get('sample_id')} would show rule-engine forecast "
            f"contamination at {leaked[:10]}; these state or hint at the answer")
    used = _tokens(view)
    if used > budget:
        raise ValueError(f"prompt view is {used} tokens, over the {budget} budget; "
                         "an audit layer was probably reintroduced")
    view["view_meta"] = {"approx_tokens": used, "token_budget": budget,
                         "dropped": ["profile_snapshot.evidence_summary",
                                     "profile_snapshot.presentation",
                                     "profile_snapshot rule-forecast fields and prose",
                                     "observation_coverage (label-side fields)"],
                         "dropped_reason": ("audit layers, restated evidence; "
                                            "rule-engine forecast fields/prose and "
                                            "label-side coverage state the answer")}
    return view
