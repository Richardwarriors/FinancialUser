"""NAACL27 fork configuration: profile dimensions and the time boundary.

What this fork changes relative to the production harness
---------------------------------------------------------
1. **Point-in-time boundary.** The production pipeline reads every row a user
   has ever produced, so a profile built today absorbs everything up to today.
   That is correct for production and fatal for a prediction benchmark: the
   profile would contain the behaviour it is supposed to predict. Here the
   boundary is enforced in the *database* (see `load_naacl_db.py`), so every
   consumer — including ones that read a whole table — only ever sees rows on or
   before `PROFILE_CUTOFF`. No metric in `app/` is computed against wall-clock
   time, so this is sufficient rather than merely convenient.

2. **Five dimensions instead of ten.** The tree keeps only the L1 blocks that
   the behavior_events/trade_executions/account_snapshots streams can actually support:

       risk_preference          what risk the user takes (revealed, not stated)
       emotion_and_psychology   disposition effect, biases, behavioural signals
       investment_logic         style, research depth, portfolio belief
       app_behavior_analysis    time/attention, engagement, attention vs capital
       investor_assessment      experience, account structure, allocation, risk

   Dropped, and why:

       user_life_cycle          needs age / career / life stage — not in financial data data
       background               needs jurisdiction, tax status, income — not in financial data data
       goals                    needs a questionnaire — not in financial data data
       constraints              needs liquidity needs, insurance, debt — not in financial data data
       personalization_and_rules  service preferences, not an investment trait

   Dropping these is not a loss of information: 20 of the 60 KYC features are
   already classed `unavailable` in `kyc_catalog.py`, and they concentrate in
   exactly these blocks. The remaining blocks carry the evidence.

3. **Three KYC blocks instead of seven.** The kept dimensions reference only
   `risk_and_protection`, `investment_logic` and `portfolio_state`. The other
   four KYC blocks exist to feed dimensions we no longer render.

To revert to the production behaviour, set `KEEP_L1 = None` and
`KEEP_KYC_L1 = None`.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Time boundary. Must match the cutoff passed to `load_naacl_db.py`.
# 2026-08-01 is the 20th day of trade_executions's global window (2026-07-13 onwards), so
# the profile is built from the first 20 days and prediction starts after them.
# --------------------------------------------------------------------------
PROFILE_CUTOFF = "2026-08-01"

# --------------------------------------------------------------------------
# Profile tree dimensions. Order is the presentation order.
# --------------------------------------------------------------------------
KEEP_L1 = [
    "risk_preference",
    "emotion_and_psychology",
    "investment_logic",
    "app_behavior_analysis",
    "investor_assessment",
]

# KYC blocks the kept dimensions resolve against (see L1_DEFS `("kyc", ...)` sources).
KEEP_KYC_L1 = [
    "risk_and_protection",
    "investment_logic",
    "portfolio_state",
]

# Summary agent grouping. The production tree uses 4 calls for 10 dimensions; five
# dimensions need two, and grouping keeps the related ones together so a summary
# can make a claim that spans them (e.g. risk appetite vs realised drawdown).
KEEP_SUMMARY_GROUPS = [
    ["risk_preference", "emotion_and_psychology"],
    ["investment_logic", "app_behavior_analysis", "investor_assessment"],
]

# L2 labels that are structurally unanswerable from financial data data. They stay in the
# definition (so the gap is visible and countable) but are excluded from the
# summary prompt, matching the production rule that a gap must never be handed
# to a model — describing what is missing is how an unknown becomes a finding.
DROP_UNSUPPORTED_L2_IN_PROMPT = True
