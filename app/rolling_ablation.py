"""Information ablations over the prompt view, for causal attribution.

Why this operates on the prompt VIEW and not on the sample
----------------------------------------------------------
`build_prompt_view` raises when `recent_7d` or `market_7d` is falsy, so an
ablation cannot delete a top-level key from the sample: the run would stop
instead of producing a prompt.  Ablating the view instead keeps the projection
contract intact and keeps the ablation auditable -- the archived sample is
untouched, and the arm is reproducible from its name alone.

Why a marker and not a deletion
-------------------------------
A silently missing `recent_7d.trades` is not the absence of information, it is
the false statement "no trades occurred".  That would inject evidence rather
than remove it, and the measured drop would confound the two.  Every ablated
node is therefore replaced by an explicit withheld marker, so the model is told
that the block exists and was not shown to it.

What an arm can and cannot claim
--------------------------------
The profile is a long-horizon narrative built from the same logs.  Measured over
25 cohort samples, its prose and traits restate holdings in 25/25, trading habit
in 25/25, app behaviour in 15/25 and index regime in 6/25.  So an arm removes
the RECENT, POINT-IN-TIME block, not the model's knowledge of that source.  Read
a drop as "the recent concrete window mattered beyond what the profile already
said".  The `profile` arm is the counterpart that removes the habitual layer.
"""
from __future__ import annotations

import copy

# arm -> dotted paths inside the prompt view
ABLATIONS: dict[str, tuple[str, ...]] = {
    "market": (
        "market_7d",
    ),
    "holdings": (
        "recent_7d.ticker_candidates.current_holdings",
        "recent_7d.portfolio_changes",
        "market_7d.portfolio",
        # _movers_block iterates the holdings book, so `movers` names the held
        # tickers. Leaving it in would leak the holdings the arm claims to remove.
        "market_7d.movers",
    ),
    "behavior7d": (
        "recent_7d.behavior",
        "recent_7d.ticker_candidates.recently_viewed",
        # _watchlist_block carries a per-ticker `views` count, which is app
        # behaviour priced up -- the same signal the arm removes from recent_7d.
        "market_7d.watchlist",
    ),
    "trades": (
        "recent_7d.trades",
        "recent_7d.trade_activity",
        "recent_7d.ticker_candidates.recently_traded",
    ),
    "profile": (
        "profile.headline",
        "profile.summary",
        "profile.dimensions",
        "profile.traits",
    ),
}

# Declared, not hidden: `market_7d.portfolio` is priced from the holdings book,
# so the `market` and `holdings` arms overlap on it. `market` removes it as part
# of the whole market block; `holdings` removes it as holdings-derived. Their
# effects are therefore not additive and must not be summed.
OVERLAPS = {
    ("market", "holdings"): ("market_7d.portfolio", "market_7d.movers"),
    ("market", "behavior7d"): ("market_7d.watchlist",),
    # Observed, not theorised. `recent_7d.portfolio_changes` is a boundary-book
    # delta, so it is holdings information that also proves a trade happened. On
    # the first smoke sample the `trades` arm still answered "Repeated recent
    # adds to RKLB, SOXL, and MU" -- it recovered the trading it had been denied
    # from the book deltas. The `trades` arm therefore measures the loss of the
    # EXPLICIT trade log, not of all evidence that trading occurred.
    ("holdings", "trades"): ("recent_7d.portfolio_changes",),
}

ARMS = ("full",) + tuple(ABLATIONS)


def _withheld(arm: str, path: str) -> dict:
    return {"withheld_for_ablation": True, "arm": arm, "path": path,
            "note": "This block exists in the record but was not shown to you. "
                    "Treat it as unknown, not as zero or as absence of activity."}


def apply_ablation(view: dict, arm: str) -> tuple[dict, list[str]]:
    """Return (ablated copy of `view`, paths actually replaced).

    Raises on an unknown arm, and on a path that is absent from the view --
    a silently-missing path would make the arm weaker than its name claims.
    """
    if arm == "full":
        return view, []
    if arm not in ABLATIONS:
        raise ValueError(f"unknown ablation arm {arm!r}; known: {sorted(ARMS)}")
    out = copy.deepcopy(view)
    hit: list[str] = []
    for path in ABLATIONS[arm]:
        parts = path.split(".")
        node = out
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f"ablation {arm!r}: path {path!r} missing at {p!r}")
            node = node[p]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"ablation {arm!r}: path {path!r} missing at {leaf!r}")
        node[leaf] = _withheld(arm, path)
        hit.append(path)
    return out, hit
