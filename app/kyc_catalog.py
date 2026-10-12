"""KYC feature catalog: the 7 L1 x L2 x L3 tree, with honest coverage metadata.

This module is deliberately the *only* place that decides what a feature means
and whether the data can support it. Three coverage classes:

  deterministic  computable from the financial data streams alone (no LLM, auditable)
  llm            a judgement call over real evidence (must cite evidence refs)
  unavailable    requires data we do not have (KYC questionnaire, account
                 opening data, income/debt, insurance, jurisdiction)

`unavailable` features are still emitted — with `insufficient_data`, an explicit
`data_needed` list and no invented value. That gap list is itself the useful
product for an advisor: it is the questionnaire they still have to run.

A fourth class sits between the two:

  proxy  we do not have the fact, but the data supports a defensible indirect
         estimate or a bounded statement. Emitted with `is_proxy=True`, the
         `proxy_basis` it was derived from, and a `caveat` stating what it is
         NOT — so it can never be mistaken for the real KYC fact.

Nothing here may ever guess. A profile that "fills in" annual income from
browsing behaviour is worse than one that says it does not know.
"""
from .taxonomy import SECTOR_MAP, ticker_info

# coverage: D = deterministic, L = LLM over evidence, P = proxy, X = unavailable from our data
D, L, P, X = "deterministic", "llm", "proxy", "unavailable"

# streams that exist in UserFinancialSnapshot
behavior_events = "behavior_events"      # in-app behaviour events
trade_executions = "trade_executions"      # executed trades
account_snapshots = "account_snapshots"      # daily account snapshot (assets + holdings)
conversations = "conversations"      # conversation titles
STATED = "stated"    # users.base_profile_summary (self-report)

# key -> (label_zh, coverage, streams, data_needed[])
CATALOG = {
    "goals_and_constraints": {
        "label": "Objectives and constraints",
        "l2": {
            "goal_definition": {
                "label": "Target definition",
                "features": {
                    "primary_financial_goal": ("Primary financial objective", X, [], ["KYC questionnaire: goal type (retirement/education/home purchase/wealth appreciation, etc.)"]),
                    "goal_priority_order": ("Objective priorities", X, [], ["KYC questionnaire: multi-objective ranking"]),
                    "target_amount": ("Target amount", X, [], ["KYC questionnaire: target amount"]),
                    "target_date": ("Target deadline date", X, [], ["KYC questionnaire: goal achievement timeline"]),
                },
            },
            "funding_plan": {
                "label": "Capital plan",
                "features": {
                    "planned_contribution": ("Planned investment", P, [account_snapshots],
                                            ["KYC questionnaire: regular investment plan; currently the observed deposit cadence"]),
                    "planned_withdrawal": ("Planned withdrawal", P, [account_snapshots],
                                          ["KYC questionnaire: withdrawal arrangements; currently the observed withdrawal cadence"]),
                    "investment_horizon_framework": ("Investment horizon framework", L, [account_snapshots, behavior_events, STATED],
                                                     ["Account holding period", "Position turnover", "Chart-viewing period distribution"]),
                },
            },
            "flexibility": {
                "label": "Flexibility",
                "features": {
                    "goal_flexibility": ("Target adjustment room", X, [], ["KYC questionnaire: whether goals can be deferred/reduced"]),
                },
            },
            "liquidity": {
                "label": "Liquidity",
                "features": {
                    "minimum_liquidity_reserve": ("Lowest liquidity reserve", L, [account_snapshots],
                                                  ["Cash proportion is an observation proxy, not a user-declared reserve requirement"]),
                },
            },
        },
    },
    "risk_and_protection": {
        "label": "Risk and protection",
        "l2": {
            "willingness": {
                "label": "Risk willingness",
                "features": {
                    "confirmed_risk_willingness": ("Confirmed risk willingness", X, [], ["KYC questionnaire or risk assessment: user confirmation required"]),
                    "maximum_uncomfortable_drawdown": ("Maximum discomfort drawdown", L, [account_snapshots, trade_executions],
                                                       ["Drawdown actually endured is indirect evidence, not a user-stated threshold"]),
                },
            },
            "capacity": {
                "label": "Risk tolerance",
                "features": {
                    "maximum_affordable_loss": ("Maximum tolerable loss", P, [account_snapshots, trade_executions],
                                                ["KYC questionnaire: tolerable loss amount; currently proxied by the lower bound of the actually experienced drawdown range"]),
                    "risk_capacity": ("Risk tolerance", L, [account_snapshots, behavior_events],
                                      ["Investable asset scale and time horizon are proxy variables, not suitability conclusions"]),
                },
            },
            "required_risk": {
                "label": "Risk required for target",
                "features": {
                    "required_return_to_goal": ("Return required to achieve the target", X, [], ["Target amount and deadline (see goals_and_constraints)"]),
                },
            },
            "scope": {
                "label": "Applicable scope",
                "features": {
                    "account_or_goal_specific_risk_level": ("Account/goal-level risk level", P, [account_snapshots],
                                                            ["Account-level data is not split by target; currently an applicability scope statement"]),
                },
            },
            "protection": {
                "label": "Coverage",
                "features": {
                    "health_insurance_coverage": ("Health insurance coverage", X, [], ["KYC questionnaire: insurance coverage"]),
                    "life_disability_coverage": ("Life insurance/disability insurance coverage", X, [], ["KYC questionnaire: insurance coverage"]),
                    "protection_gap_assessment": ("Coverage gap assessment", X, [], ["Insurance information + family responsibility information"]),
                },
            },
        },
    },
    "portfolio_state": {
        "label": "Current Portfolio Status",
        "l2": {
            "account": {
                "label": "Account",
                "features": {
                    "account_types_and_tax_wrappers": ("Account type and tax structure", X, [], ["Account opening information (taxable/retirement accounts, etc.)"]),
                    "account_balances_and_cash": ("Account balance and cash", D, [account_snapshots], []),
                },
            },
            "allocation": {
                "label": "Asset allocation",
                "features": {
                    "asset_class_allocation": ("Asset category allocation", D, [account_snapshots], []),
                    "strategy_bucket_structure": ("Strategy bucket structure", D, [account_snapshots], []),
                },
            },
            "concentration": {
                "label": "Concentration",
                "features": {
                    "largest_position_concentration": ("Maximum position concentration", D, [account_snapshots], []),
                    "sector_concentration": ("Industry concentration", D, [account_snapshots], []),
                    "employer_stock_concentration": ("Employer stock concentration", P, [account_snapshots],
                                                     ["No employer information; currently proxied by the equivalent single-holding concentration risk pattern"]),
                },
            },
            "exposure": {
                "label": "Exposure",
                "features": {
                    "geographic_and_currency_exposure": ("Geographic and currency exposure", P, [account_snapshots],
                                                         ["No currency data; currently proxied by the instrument issuer's place of registration, with unlisted instruments counted as U.S. stocks"]),
                    "illiquid_asset_exposure": ("Illiquid asset exposure", P, [account_snapshots],
                                                ["Can only state that no such assets are within the visible scope; assets outside the platform are not visible"]),
                    "leverage_and_derivatives_exposure": ("Leverage and derivatives exposure", D, [account_snapshots, trade_executions, behavior_events], []),
                },
            },
            "cashflow_and_experience": {
                "label": "Cash flow and experience",
                "features": {
                    "contribution_and_withdrawal_pattern": ("Deposit and withdrawal pattern", D, [account_snapshots], []),
                    "drawdown_experience": ("Drawdown experience", D, [account_snapshots], []),
                },
            },
        },
    },
    "investment_logic": {
        "label": "Investment logic",
        "l2": {
            "style": {
                "label": "Style",
                "features": {
                    "preferred_investment_style": ("Preferred investment style", L, [behavior_events, trade_executions, account_snapshots, STATED], []),
                },
            },
            "fundamental_research": {
                "label": "Research depth",
                "features": {
                    "fundamental_analysis_depth": ("Depth of fundamental analysis", L, [behavior_events], []),
                    "investment_domain_expertise": ("Professionalism in investment fields", L, [behavior_events, conversations], []),
                    "thesis_evidence_requirement": ("Evidentiary requirements", L, [behavior_events], ["The relationship from research to order placement is indirectly inferred"]),
                    "valuation_sensitivity": ("Valuation sensitivity", L, [behavior_events], ["Can only be inferred indirectly from viewing valuation-related pages"]),
                },
            },
            "portfolio_belief": {
                "label": "Portfolio conviction",
                "features": {
                    "diversification_belief": ("Diversification belief", L, [behavior_events, account_snapshots], []),
                },
            },
            "resource": {
                "label": "Invested resources",
                "features": {
                    "time_and_attention_budget": ("Time and attention allocation", L, [behavior_events], []),
                },
            },
        },
    },
    "advisory_preferences": {
        "label": "Financial advisor preferences",
        "l2": {
            "authority": {
                "label": "Decision authorization",
                "features": {
                    "advisor_autonomy_preference": ("Financial advisor autonomy preference", L, [conversations, behavior_events], ["No direct inquiry data"]),
                },
            },
            "copy": {
                "label": "Communication preferences",
                "features": {
                    "advisory_language_preference": ("Language preference", L, [conversations], []),
                    "explanation_depth_preference": ("Preference for explanation depth", L, [conversations], []),
                    "communication_tone_preference": ("Preferred communication tone", L, [conversations], ["Title samples are usually too few"]),
                    "output_format_preference": ("Output format preference", L, [conversations], []),
                },
            },
            "service_rhythm": {
                "label": "Service cadence",
                "features": {
                    "notification_trigger_preferences": ("Notification trigger preferences", X, [], ["No notification settings data"]),
                    "portfolio_review_cadence": ("Portfolio review frequency", L, [behavior_events, account_snapshots], ["Access cadence is a proxy variable"]),
                },
            },
        },
    },
    "financial_capacity": {
        "label": "Financial capacity",
        "l2": {
            "assets": {
                "label": "Assets",
                "features": {
                    "investable_assets_band": ("Investable asset range", D, [account_snapshots], []),
                    "external_asset_coverage": ("External asset coverage", P, [account_snapshots],
                                                ["Can only state the visible scope; assets outside the platform are not visible"]),
                },
            },
            "income": {
                "label": "Income",
                "features": {
                    "annual_income_band": ("Annual income range", P, [account_snapshots],
                                           ["KYC questionnaire: income; currently proxied by the lower bound of disposable surplus, not an income conclusion"]),
                    "income_stability": ("Income stability", P, [account_snapshots],
                                         ["KYC questionnaire: occupation and income; currently using deposit regularity as a weak proxy"]),
                    "expected_income_change": ("Expected income change", X, [], ["KYC questionnaire: career planning"]),
                },
            },
            "cashflow": {
                "label": "Cash flow",
                "features": {
                    "essential_expense_band": ("Necessary expense range", X, [], ["KYC questionnaire: expenses"]),
                    "emergency_fund_runway": ("Number of months emergency funds can support", P, [account_snapshots],
                                              ["KYC questionnaire: expenses; currently estimated at the lower bound of account cash ÷ observed net outflows"]),
                },
            },
            "debt": {
                "label": "Liabilities",
                "features": {
                    "total_debt_band": ("Total liabilities range", P, [account_snapshots],
                                        ["KYC questionnaire: liabilities; currently can only detect financing liabilities within the account"]),
                    "debt_service_ratio": ("Debt service ratio", X, [], ["KYC questionnaire: liabilities and repayments"]),
                },
            },
        },
    },
    "jurisdiction_and_lifecycle": {
        "label": "Jurisdiction and lifecycle",
        "l2": {
            "jurisdiction": {
                "label": "Jurisdiction",
                "features": {
                    "legal_residence": ("Legal residence", X, [], ["KYC account opening information"]),
                    "tax_residency": ("Tax residency status", X, [], ["KYC account opening information"]),
                    "cross_border_status": ("Cross-border status", X, [], ["KYC account opening information"]),
                },
            },
            "career": {
                "label": "Occupation",
                "features": {
                    "employment_status": ("Employment status", X, [], ["KYC questionnaire: occupational information"]),
                    "occupation_type": ("Occupation Type", X, [], ["KYC questionnaire: occupational information"]),
                    "occupation_industry": ("Industry", L, [STATED, conversations, behavior_events],
                                            ["Can only make a determination when the industry is explicitly mentioned in the user's self-report or dialogue"]),
                },
            },
            "lifecycle": {
                "label": "Lifecycle",
                "features": {
                    "wealth_lifecycle_stage": ("Wealth lifecycle stage", L, [account_snapshots, behavior_events, STATED], []),
                },
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Rolling profile configuration: keep only the KYC blocks the retained profile dimensions
# resolve against. `iter_features` below is the single accessor every consumer
# goes through (kyc_tree, kyc_agents, catalog_summary), so filtering the catalog
# here removes the blocks everywhere at once.
#
# The dropped four blocks are not a loss of measurable information: they are the
# ones whose features are dominated by `unavailable` (goals 5/9, jurisdiction
# 5/7, financial_capacity 3/9, advisory_preferences 1/7 unavailable, and the
# rest questionnaire-dependent). See app/rolling_config.py.
# ---------------------------------------------------------------------------
try:
    from .rolling_config import KEEP_KYC_L1
except ImportError:                                    # standalone import
    from rolling_config import KEEP_KYC_L1

if KEEP_KYC_L1:
    _keep_kyc = set(KEEP_KYC_L1)
    _unknown = _keep_kyc - set(CATALOG)
    assert not _unknown, f"KEEP_KYC_L1 names unknown blocks: {sorted(_unknown)}"
    _dropped = sorted(set(CATALOG) - _keep_kyc)
    CATALOG = {k: v for k, v in CATALOG.items() if k in _keep_kyc}
    DROPPED_KYC_L1 = _dropped
else:
    DROPPED_KYC_L1 = []


def iter_features():
    """Yield (l1_key, l2_key, l3_key, label, coverage, streams, data_needed)."""
    for l1, node in CATALOG.items():
        for l2, l2node in node["l2"].items():
            for l3, spec in l2node["features"].items():
                label, cov, streams, needed = spec
                yield l1, l2, l3, label, cov, streams, needed


def build_llm_features():
    """L3 keys the LLM is asked to judge, grouped by L1 (used by kyc_agents)."""
    out = {}
    for l1, l2, l3, label, cov, streams, needed in iter_features():
        if cov == L:
            out.setdefault(l1, []).append(l3)
    return out


def catalog_summary():
    from collections import Counter
    cov = Counter()
    per_l1 = {}
    total = 0
    for l1, l2, l3, label, c, streams, needed in iter_features():
        total += 1
        cov[c] += 1
        per_l1.setdefault(l1, Counter())[c] += 1
    return {"total": total, "coverage": dict(cov), "per_l1": {k: dict(v) for k, v in per_l1.items()},
            "streams_used": sorted({s for *_x, streams, _n in iter_features() for s in streams})}
