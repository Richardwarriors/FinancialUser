"""Plain-language glossary and time window for every metric shown to a user.

Why this exists
---------------
A reader cannot act on `loss_attention_ratio = 9.33`. Two things are missing:

1. **meaning** — what the number is about, in words;
2. **window** — which days it was computed from.

The window is the one that matters most for trust. A figure labelled only "90 天"
invites the reader to assume a different period than the one actually used, and
several metrics here deliberately use DIFFERENT windows from each other
(turnover 7 days, monthly turnover 28 days, holding weeks ~90 days, trade stats
52 days). Presenting them side by side with no window is how a correct number
becomes a wrong conclusion.

Design
------
* `ENTRIES` owns the MEANING. It is static text, keyed by metric path.
* `window_for()` owns the WINDOW, derived from the profile at read time — never
  hardcoded, because the real window travels with the data (`window`,
  `observation_days`, `date_range`, `as_of`).
* `annotations()` merges the two for the viewer.

Nothing here is stored: annotations are derived on read, so a glossary edit reaches
every profile immediately without a data migration.

A missing entry is NOT an error and must not be silent either — `annotations()`
reports coverage so `tests/test_metric_glossary.py` can fail when a metric gains no
explanation.
"""
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Meaning
# ---------------------------------------------------------------------------
# `meaning` is written for the person reading their own profile: no field names,
# no jargon. `how` is the arithmetic, so the number can be checked. `limit` states
# what the metric cannot support — the honesty contract, per-metric.
ENTRIES = {
    # ---- 账户与收益 ----------------------------------------------------------
    "asset.equity_change_pct": {
        "label": "Total return",
        "meaning": "Change in account net asset value at the end of the observation window relative to the start, after excluding the impact of deposits and withdrawals.",
        "how": "Use the time-weighted equity curve: ending equity ÷ starting equity − 1, with cash inflows and outflows removed from the curve.",
        "unit": "%",
    },
    "asset.last_equity": {
        "label": "Period-end net assets",
        "meaning": "Total account assets on the last day of the observation window (cash + securities market value).",
        "how": "Take total assets on the last trading day of the equity curve.",
        "unit": "Yuan",
    },
    "asset.max_drawdown": {
        "label": "Maximum drawdown",
        "meaning": "The maximum decline from any historical high to the subsequent low during the observation window, measuring the account's worst stage.",
        "how": "Daily cumulative maximum of the net asset value curve, taking the minimum of (current value ÷ cumulative high − 1).",
        "unit": "%",
        "limit": "Deepest historical decline, does not represent future risk.",
    },
    "asset.equity_volatility": {
        "label": "Daily volatility",
        "meaning": "The magnitude of daily fluctuations in account net asset value (standard deviation); higher means less stable net asset value.",
        "how": "Sample standard deviation of daily net asset value returns. Not annualized.",
        "unit": "%",
    },
    "asset.return_reliable": {
        "label": "Whether the return definition is usable",
        "meaning": "Whether net asset value curve coverage is sufficient, determining whether returns and drawdown can serve as conclusions.",
        "how": "Coverage days and completeness must meet the threshold to be marked as usable.",
        "unit": None,
    },
    "risk_exposure.sharpe_ratio": {
        "label": "Rough Sharpe",
        "meaning": "The return earned per unit of volatility assumed, roughly measuring whether this volatility is worth it.",
        "how": "Average daily return ÷ daily volatility, with the risk-free rate set to 0, not annualized.",
        "unit": None,
        "limit": "The definition is rough and not annualized; suitable only for cross-sectional comparison of observation-window characteristics within this page, and not for use as a standard Sharpe ratio.",
    },
    "holding_endurance.avg_holding_weeks": {
        "label": "Average holding weeks",
        "meaning": "Average holding duration per position, indicating whether the strategy is short-term or long-term.",
        "how": "Observation weeks ÷ average holdings is an approximate inverse of turnover speed.",
        "unit": "Week",
    },
    "weekly_pnl": {
        "label": "Position unrealized profit/loss",
        "meaning": "Total unrealized profit or loss of current positions relative to cost.",
        "how": "For each holding, sum (latest price − cost price) × quantity.",
        "unit": "Yuan",
        "limit": "Counts only instruments still currently held; excludes gains and losses from exited positions. Fees and taxes are not deducted.",
    },
    "weekly_return_pct": {
        "label": "Account weekly return",
        "meaning": "Equity change rate for the entire account over the past week, including exited and realized portions.",
        "how": "Take the recent one-week range from the time-weighted equity curve.",
        "unit": "%",
        "limit": "Uses a different definition from \"unrealized P&L\" (one looks only at current holdings, the other at the entire account); inconsistent directions are normal.",
    },

    # ---- 中期 ---------------------------------------------------------------
    "market_value_trend.avg_mv_4w": {
        "label": "Average market value",
        "meaning": "Average position market value over the past 4 weeks.",
        "how": "Mean daily securities market value within the window.",
        "unit": "Yuan",
    },
    "market_value_trend.mv_change_4w_pct": {
        "label": "Market value change",
        "meaning": "Change ratio of position market value over the past 4 weeks, affected by both buying and selling and price movements.",
        "how": "Window-end market value ÷ window-start market value − 1.",
        "unit": "%",
        "limit": "Market value change cannot distinguish between \"active position additions/reductions\" and \"price fluctuations\"; it needs to be viewed together with turnover.",
    },
    "market_value_trend.mv_volatility_4w": {
        "label": "Market value volatility",
        "meaning": "Intraday fluctuation range of position market value over the past 4 weeks.",
        "how": "Sample standard deviation of daily market value changes.",
        "unit": "%",
    },
    "pnl_trend.cum_pnl_4w": {
        "label": "Cumulative P&L",
        "meaning": "Cumulative unrealized profit and loss of current positions over the past 4 weeks.",
        "how": "Cumulative unrealized P&L of current holdings within the window.",
        "unit": "Yuan",
    },
    "position_habit.avg_positions_4w": {
        "label": "Average number of holdings",
        "meaning": "Average number of instruments held simultaneously over the past 4 weeks.",
        "how": "Mean daily number of holdings within the window.",
        "unit": "Only",
    },
    "position_habit.avg_leverage_4w": {
        "label": "Average leverage",
        "meaning": "Proportion of total assets represented by securities market value; the closer it is to 1, the less cash is held.",
        "how": "Mean of daily securities market value ÷ total assets.",
        "unit": None,
    },
    "average_down_tendency": {
        "label": "Tendency to average down",
        "meaning": "Strength of the tendency to increase positions while in an unrealized loss; a stronger tendency indicates a greater inclination to \"buy more as it falls.\"",
        "how": "Frequency of adding to positions after they show an unrealized loss.",
        "unit": None,
        "limit": "This is an observation of behavioral tendencies, not evidence of an intentional strategy; increasing a position may also be part of an additional-purchase plan.",
    },
    "pyramid_winner_tendency": {
        "label": "Tendency to add to profitable positions",
        "meaning": "Strength of the tendency to continue increasing positions while in an unrealized gain.",
        "how": "Frequency of adding to positions after they show an unrealized gain.",
        "unit": None,
    },

    # ---- 短期 ---------------------------------------------------------------
    "portfolio_snapshot.total_market_value": {
        "label": "Total position market value",
        "meaning": "Total market value in the securities account on the most recent trading day with positions.",
        "how": "Sum of the market values of all instruments in the latest position snapshot.",
        "unit": "Yuan",
    },
    "portfolio_snapshot.num_positions": {
        "label": "Number of held instruments",
        "meaning": "Number of instruments held on the most recent trading day with positions.",
        "how": "Number of instruments in the latest position snapshot.",
        "unit": "Only",
    },
    "short_term_trading.total_orders_7d": {
        "label": "Number of trades",
        "meaning": "Number of orders actually executed in the last 7 days.",
        "how": "Count successfully executed orders by actual execution time.",
        "unit": "transactions",
    },
    "short_term_trading.net_buy_7d": {
        "label": "Net purchases over the past 7 days",
        "meaning": "Purchase amount − sale amount over the past 7 days; a positive value indicates a net increase in position.",
        "how": "Buy transaction amount minus sell transaction amount within the window.",
        "unit": "Yuan",
    },
    "short_term_trading.turnover_7d": {
        "label": "Turnover",
        "meaning": "Ratio of trading value to position market value over the past 7 days, measuring trading frequency.",
        "how": "Transaction amount within the window ÷ average holding market value.",
        "unit": None,
    },
    "medium_term_trading.monthly_turnover": {
        "label": "30-day turnover",
        "meaning": "Ratio of transaction value to position market value over the past 4 weeks.",
        "how": "Transaction amount within the window ÷ average holding market value.",
        "unit": None,
    },

    # ---- 成交与结果 ----------------------------------------------------------
    "activity.total_orders": {
        "label": "Total number of transactions",
        "meaning": "Total number of orders actually executed during the observation window.",
        "how": "Count successfully executed orders by actual execution time.",
        "unit": "transactions",
    },
    "activity.trading_days": {
        "label": "Number of trading days",
        "meaning": "Number of days during the observation window with at least one executed trade.",
        "how": "Count distinct trading days.",
        "unit": "Days",
    },
    "execution.avg_trade_size": {
        "label": "Average transaction amount",
        "meaning": "Average amount per transaction, reflecting bet size.",
        "how": "Total trade value ÷ number of trades.",
        "unit": "Yuan",
    },
    "execution.matched_round_trips": {
        "label": "Number of closed positions",
        "meaning": "Number of complete rounds that can be paired as \"buy -> sell\", serving as the denominator for win-rate metrics.",
        "how": "Pair buys and sells using FIFO; count only successfully paired transactions.",
        "unit": "transactions",
        "limit": "Sell shortages or purchases not followed by a sale cannot be paired, so this number is usually much lower than the total number of executed transactions.",
    },
    "execution.avg_hold_days": {
        "label": "Average holding days",
        "meaning": "Average holding days from purchase to sale among matched round trips.",
        "how": "Mean holding days across rounds.",
        "unit": "Days",
    },
    "results.win_rate": {
        "label": "Win Rate",
        "meaning": "Proportion of profitable round trips among matched round trips.",
        "how": "Number of profitable rounds ÷ number of balanced rounds.",
        "unit": "%",
        "limit": "The denominator includes only balanced rounds. The current sample is very small; this ratio is for reference only and does not represent overall trading performance.",
    },
    "results.profit_factor": {
        "label": "Profit-to-loss ratio",
        "meaning": "Total profit ÷ total loss, greater than 1 indicates overall profits exceed losses.",
        "how": "Total amount of profitable rounds ÷ total amount of losing rounds.",
        "unit": None,
        "limit": "Extremely unstable with few samples; 0 when there are no profitable rounds.",
    },
    "results.expectancy_per_trade": {
        "label": "Expected P&L per round",
        "meaning": "Average profit or loss amount per completed round trip.",
        "how": "Total realized profit and loss ÷ number of matched round trips.",
        "unit": "Yuan",
    },
    "results.total_realised_pnl": {
        "label": "Realized profit and loss",
        "meaning": "Total actual profit and loss from portions sold.",
        "how": "Sum of profit and loss from matched round trips (before fees and taxes).",
        "unit": "Yuan",
    },

    # ---- 持仓动态 -----------------------------------------------------------
    "rotation.composition_change_rate": {
        "label": "Portfolio Rotation Intensity",
        "meaning": "How much the position structure changes each day; higher values indicate more frequent rebalancing.",
        "how": "Difference ratio between positions on adjacent days, with a representative value taken for the window.",
        "unit": None,
    },
    "avg_similarity_day_to_day": {
        "label": "Similarity of positions on adjacent days",
        "meaning": "Degree of overlap between positions on adjacent days; a higher value indicates a more stable portfolio.",
        "how": "Similarity score of position weights on adjacent days, averaged.",
        "unit": None,
    },
    "conviction.max_position_weight_ever": {
        "label": "Maximum single-instrument position",
        "meaning": "Highest share of a single instrument observed during the observation window, reflecting the degree of bet concentration.",
        "how": "Calculate the proportion for each instrument daily and take the maximum.",
        "unit": "%",
    },
    "reentry_rate": {
        "label": "Re-entry rate",
        "meaning": "Proportion of Buying the Same Instrument Again After Exiting.",
        "how": "Number of re-entries ÷ number of exits.",
        "unit": "%",
    },
    "usable_ratio": {
        "label": "Position data availability ratio",
        "meaning": "Share of days during the observation window with complete holding data directly usable for calculation.",
        "how": "Valid days ÷ observation window days.",
        "unit": "%",
        "limit": "Missing days carry over the previous day's holdings and are not counted as exits, so this ratio directly determines the reliability of subsequent metrics.",
    },
    "carried_forward_days": {
        "label": "Days continuing the previous day's position",
        "meaning": "Number of days with missing position data, replaced with the previous day's position.",
        "how": "Count by the number of trading days with missing holding snapshots.",
        "unit": "Days",
        "limit": "These days are excluded from starting-a-new-position or exit statistics; otherwise missing data could be misread as an exit.",
    },

    # ---- 关注 vs 资金 --------------------------------------------------------
    "held_ticker_view_ratio": {
        "label": "Proportion of followed instruments held",
        "meaning": "Proportion of all instrument-level behavioral events occurring on currently held instruments. High = mainly monitoring own positions.",
        "how": "Events on currently held instruments ÷ total events attributable to instruments.",
        "unit": "%",
    },
    "nonheld_ticker_view_ratio": {
        "label": "Proportion of followed instruments not held",
        "meaning": "Proportion of all instrument-level behavioral events occurring on instruments not held. High = mainly seeking new opportunities.",
        "how": "Events on non-held instruments ÷ total events attributable to instruments.",
        "unit": "%",
    },
    "prior_attention_rate_7d": {
        "label": "Research in the 7 days before starting a new position",
        "meaning": "Among observed position starts, how many actually viewed the instrument within 7 days before buying. Measures how common \"research first, then buy\" is.",
        "how": "Number of position starts with behavioral events for this instrument in the 7 days before starting a new position ÷ observable position starts.",
        "unit": "%",
        "limit": "Covers only \"viewed before buying\". Positions started without prior attention are included neither in the numerator nor in the denominator's preference interpretation.",
    },
    "research_to_holding_conversion": {
        "label": "Attention→position initiation conversion rate",
        "meaning": "Proportion of non-held instruments reaching research-grade that were ultimately purchased.",
        "how": "Number of research-grade instruments subsequently used to start a new position ÷ number of research-grade non-held instruments.",
        "unit": "%",
        "limit": "The denominator includes only instruments not held at the start of the observation window (positions held on the first day cannot be used to determine the position-building time). This ratio is not provided when the sample is insufficient.",
    },
    "attention_on_losing_positions": {
        "label": "Attention volume during unrealized losses",
        "meaning": "Number of user view events for held instruments while they were in an unrealized loss.",
        "how": "Count after classification by the instrument's actual profit/loss status on the event date.",
        "unit": "Times",
        "limit": "Counts only events on days when the instrument was actually held.",
    },
    "attention_on_winning_positions": {
        "label": "Attention Count During Unrealized Profit",
        "meaning": "Number of user view events for held instruments while they were in an unrealized gain.",
        "how": "Count after classification by the instrument's actual profit/loss status on the event date.",
        "unit": "Times",
    },
    "loss_attention_ratio": {
        "label": "Attention ratio (loss/profit)",
        "meaning": "Attention volume during unrealized losses ÷ attention volume during unrealized gains. Greater than 1 indicates more viewing during unrealized losses, while less than 1 indicates more viewing during unrealized gains.",
        "how": "Number of attention events under unrealized losses ÷ number of attention events under unrealized gains, attributed based on the P&L state on the day each event occurred.",
        "unit": "Times",
        "limit": ("Only shows that \"attention\" and an unrealized loss occurred simultaneously. Whether action was actually taken after viewing or not is shown below in \"Whether attention during an unrealized loss led to adding\"; sequence can be verified, motivation cannot."),
    },

    # ==== 摊平深度（v2.6 新增，修正一个会误导的标签）====
    "median_underwater_depth": {
        "label": "Averaging-down depth (median)",
        "meaning": "Average loss at which the position was increased while in an unrealized loss. A higher value indicates more of a \"buying more only after a larger decline\" tendency.",
        "how": "Median of |latest price/cost price − 1| across all add-to-position events at a loss.",
        "unit": "%",
        "limit": ("This is the unrealized loss at the time of increasing the position, not the final loss on that transaction. Position increases with a magnitude below 5% are nearly indistinguishable from purchases at the original price and are separately classified as shallow."),
    },
    "max_underwater_depth": {
        "label": "Averaging-down depth (maximum)",
        "meaning": "Among all add-to-position events at a loss, the deepest loss at which an addition was made.",
        "how": "Take the maximum unrealized loss magnitude among add-to-position events during unrealized losses.",
        "unit": "%",
        "limit": "A single extreme value, for reference only; median is used to assess habits.",
    },
    "shallow_underwater_rate": {
        "label": "Proportion of shallow position additions",
        "meaning": "Share of add-ons during unrealized losses with losses under 5%. A higher value indicates that most \"unrealized-loss add-ons\" are actually normal position top-ups.",
        "how": "Number of add-ons with unrealized losses < 5% ÷ total number of unrealized-loss add-ons.",
        "unit": "%",
        "limit": ("5% is the threshold because these instruments have average daily fluctuations of approximately 2%; below 5%, the adding-to-position price is nearly indistinguishable from the cost price and cannot be called \"averaging down losses\"."),
    },
    "average_down_episode_rate": {
        "label": "Averaging-down episode share",
        "meaning": "Proportion of add-position episodes classified as \"multiple adds when unrealized loss >=5%\".",
        "how": "Number of averaging-down episodes ÷ total number of add-position episodes.",
        "unit": "%",
        "limit": "Has a different denominator from \"proportion of adding to positions while at a floating loss\" (calculated per individual event); unequal values are normal.",
    },
    "average_down_episodes": {
        "label": "Number of averaging-down episodes",
        "meaning": "Number of consecutive add-position segments with floating losses >=5%.",
        "how": "Continuous additions to the same instrument (interval ≤5 days) are grouped into one segment; included when the median unrealized loss magnitude is ≥5%.",
        "unit": "items",
    },
    "shallow_underwater_episodes": {
        "label": "Number of shallow position-addition episodes",
        "meaning": "Number of add-position segments that occurred when floating losses were below 5%.",
        "how": "Grouped into segments as above, but the median unrealized loss magnitude is < 5%.",
        "unit": "items",
        "limit": "Retaining this figure is intended not to hide the fact: it shows how frequently \"adding to a position below the surface\" occurs, without calling it averaging down.",
    },

    # ==== 关注 → 行动 条件链（v2.6 新增）====
    "add_rate_underwater_with_attention": {
        "label": "Unrealized loss + viewed add-position rate",
        "meaning": "Proportion of days on which an add-on was made while the position was at an unrealized loss and the instrument had been viewed in the prior week.",
        "how": "Numerator: opportunity-days with an add on the day (≥+20% relative to the previous day); denominator: opportunity-days in a floating-loss state with a behavioral event for the instrument during the previous 7 trading days.",
        "unit": "%",
        "limit": ("The denominator is opportunity-days, not events, counted by instrument × day and therefore much larger than the event-based definition. This is a conditional frequency, not causality."),
    },
    "add_rate_underwater_without_attention": {
        "label": "Unrealized loss + not viewed add-position rate",
        "meaning": "Proportion of days on which an add-on was made while the position was at an unrealized loss and the instrument had not been viewed in the prior week.",
        "how": "As above, but there were no behavioral events for the instrument in the previous 7 trading days.",
        "unit": "%",
        "limit": "Only meaningful compared with the previous item: similar values indicate that adding to positions is unrelated to whether research was conducted.",
    },
    "add_rate_in_profit_with_attention": {
        "label": "Unrealized Profit + Viewed Add-Position Rate",
        "meaning": "Proportion of days on which an add-on was made while the position was at an unrealized gain and the instrument had been viewed in the prior week.",
        "how": "As above; P/L status is unrealized gain.",
        "unit": "%",
    },
    "add_rate_in_profit_without_attention": {
        "label": "Unrealized gain + not viewed add-on rate",
        "meaning": "Proportion of days on which an add-on was made while the position was at an unrealized gain and the instrument had not been viewed in the prior week.",
        "how": "As above; P/L status is unrealized gain, with no behavioral events in the previous 7 days.",
        "unit": "%",
    },
    # Refused-comparison view: when the comparison is not given, the page prints
    # the counts instead of the rates, so these labels need entries too.
    "opportunities_loss_with_attention": {
        "label": "Unrealized loss + viewed opportunity days",
        "meaning": ("Number of days the position was at an unrealized loss and the instrument had been viewed in the prior 7 trading days. This is only a denominator: the number of days on which an add-on was possible."),
        "how": "Count by (instrument, trading day); same-day attention is not counted as \"viewed\".",
        "unit": "Days",
        "limit": "A high number of opportunity days does not mean that adding to positions occurred—these are the two things this page distinguishes.",
    },
    "adds_loss_with_attention": {
        "label": "Unrealized loss + viewed add-position count",
        "meaning": "Actual add-position count when there is an unrealized loss and the instrument was previously viewed (numerator).",
        "how": "Number of days on which the position was increased by ≥+20% relative to the previous day during opportunity days with \"unrealized loss+viewed\".",
        "unit": "Times",
        "limit": ("When the count is too low (total <3 times), this page gives no conclusion on which side is more common and only lists these two numbers. A single action does not constitute a tendency."),
    },
    "opportunities_loss_without_attention": {
        "label": "Unrealized loss + not viewed opportunity days",
        "meaning": ("Number of days the position was at an unrealized loss and the instrument had not been viewed in the prior 7 trading days. This is the comparison denominator for the \"unrealized loss + viewed\" add-on rate."),
        "how": "Count by (instrument, trading day); no behavioral events for the instrument in the first 7 trading days.",
        "unit": "Days",
    },
    "adds_loss_without_attention": {
        "label": "Unrealized loss + not viewed add-position count",
        "meaning": "Actual add-position count when there is an unrealized loss and the instrument was not previously viewed (numerator).",
        "how": "Number of days on which the position was increased by ≥+20% relative to the previous day during opportunity days with \"unrealized loss+not viewed\".",
        "unit": "Times",
        "limit": "When the sum with the previous item is less than 3 occurrences, neither side's ratio is displayed—the denominator is sufficient, but the event count is not.",
    },
    # ==== 下一步动作推演（v2.7 新增）====
    # A forecast is the easiest thing here to misread as advice, so its entries
    # carry that limit explicitly rather than relying on the card's caveat.
    "action_forecast.primary": {
        "label": "Most likely action",
        "meaning": ("Based on this user's own behavioral habits, the highest-scoring next funding action is: start a new position / add to an existing holding / trim / exit / maintain unchanged."),
        "how": ("For each candidate action, use the action's average daily frequency as the primary signal, supplemented by its style signals (add tendency, averaging-down tendency, number of risk-reduction actions, portfolio stability, etc.) to calculate a weighted average; weights include only signals with data."),
        "unit": None,
        "limit": ("Describes the user's habits, not a market judgment or investment advice. The user may choose not to act according to these habits."),
    },
    "action_forecast.primary_score": {
        "label": "Action score",
        "meaning": "The action's relative score on a 0–1 scale; a higher score indicates that it is more likely to occur based on its historical pattern.",
        "how": ("Weighted average: Σ(weight × signal score) ÷ Σ(signal weights with weights and data). Missing signals do not participate in the denominator, so they do not lower the score."),
        "unit": None,
        "limit": ("This is a relative ranking among actions, not the probability of occurrence: a score of 0.8 does not mean there is an 80% chance of taking this action."),
    },
    "action_forecast.observed_actions": {
        "label": "Number of observed actions",
        "meaning": "Total number of position starts + adds + trims + exits actually recorded during the observation window, indicating the evidentiary basis for this analysis.",
        "how": "opens+adds+reduces+closes of position_dynamics.style.",
        "unit": "Times",
        "limit": ("The fewer the observations, the lower the confidence. Only more than 20 times is considered sufficient behavioral data; when the record is thin, the engine automatically lowers confidence."),
    },
    "investment_logic.next_action.primary_action": {
        "label": "Action determined by the model",
        "meaning": "Selection and interpretation results of engine candidate actions based on investment logic dimensions.",
        "how": ("The model can only select one action from the candidate actions already scored by the engine, or output \"unable to determine\"; incorrect selections or selections outside the candidate set will be blocked by validation."),
        "unit": None,
        "limit": ("The model must not add actions that the engine has not scored, so \"the engine says nothing is wrong, but the model says there is a major action\" will never occur here."),
    },
    "investment_logic.next_action.confidence": {
        "label": "Model confidence",
        "meaning": "The model's confidence in its determination of this action.",
        "how": "Model self-assessment, capped at engine confidence + 0.1; exceeding this is deemed invalid by validation.",
        "unit": None,
        "limit": ("The upper limit is set by the engine because the engine has already lowered confidence when evidence is insufficient; the model cannot be more confident than the engine."),
    },
    "share_on_losing": {
        "label": "Unrealized-loss attention share",
        "meaning": "Proportion of attributable attention events that occurred during an unrealized loss.",
        "how": "Unrealized loss event count ÷ (unrealized loss event count + unrealized gain event count).",
        "unit": "%",
    },
    "attention_hhi": {
        "label": "Attention concentration HHI",
        "meaning": "Whether attention is concentrated in a small number of instruments. Values closer to 1 indicate greater concentration; values closer to 0 indicate greater dispersion.",
        "how": "Sum of squared attention shares across instruments.",
        "unit": None,
    },
    "portfolio_hhi": {
        "label": "Position concentration HHI",
        "meaning": "Whether capital is concentrated in a small number of instruments. The closer to 1, the more concentrated.",
        "how": "Sum of squared position weights across instruments.",
        "unit": None,
    },
    "attended_tickers": {
        "label": "Followed instruments",
        "meaning": "Number of instruments with behavioral events during the observation window.",
        "how": "Count distinct instruments.",
        "unit": "Only",
    },
    "reached_research": {
        "label": "Reached research-grade",
        "meaning": "Number of followed instruments whose behavior reached research-level depth, rather than merely being viewed once.",
        "how": "Determine based on the highest intention stage reached for the instrument.",
        "unit": "Only",
    },
    "observable_entries": {
        "label": "Observable position initiation",
        "meaning": "The number of times changing from \"not held\" to \"held\", namely the number of new position-building events.",
        "how": "Compare holdings daily; count once when an instrument changes from absent to present.",
        "unit": "Times",
        "limit": "Instruments already held at the start of the observation window cannot be used to determine the entry time and are excluded.",
    },
    "entries_scaled_up": {
        "label": "Subsequent addition to a position",
        "meaning": "Number of times the position in this instrument was increased after starting a new position.",
        "how": "A position increase event in the same instrument after starting a new position.",
        "unit": "Times",
    },
    "entries_held_30d_plus": {
        "label": "Held for at least 30 days",
        "meaning": "Held for at least another 30 days after opening, measuring whether the purchase was intended as short-term or long-term.",
        "how": "Number of events in which a position was continuously held for more than 30 days from the opening date.",
        "unit": "Times",
    },
    "attention_to_research": {
        "label": "Attention→research",
        "meaning": "Among followed instruments, how many were researched to research-level depth.",
        "how": "Number of instruments reaching research-grade ÷ number of instruments followed.",
        "unit": "%",
        "limit": "The denominators at each stage are independent, not stepwise attrition of the same batch of instruments; therefore, the ratios cannot be multiplied sequentially.",
    },
    "research_to_entry": {
        "label": "Research→Start a new position",
        "meaning": "Number of instruments reaching research-grade that were actually purchased.",
        "how": "Number of observable position initiations ÷ number of instruments reaching research-grade coverage.",
        "unit": "%",
        "limit": "As above; the denominator differs from the previous stage, so the ratios cannot be multiplied sequentially.",
    },
    "entry_to_scale_up": {
        "label": "Start a new position → add",
        "meaning": "Proportion of cases where the position was further increased after starting a new position, measuring willingness to add after buying.",
        "how": "Number of times positions were added to after opening ÷ observable position openings.",
        "unit": "%",
        "limit": "The denominator is the number of position-building instances, not the number of instruments.",
    },
    "entry_to_persistent_holding": {
        "label": "Start a new position → continue holding",
        "meaning": "Proportion held for at least 30 days after opening, measuring willingness to hold purchases.",
        "how": "Number of new positions held for at least 30 days ÷ number of observable new positions.",
        "unit": "%",
        "limit": "Return \"—\" rather than 0 for sections with insufficient samples.",
    },
    "pre_entry_attention_7d": {
        "label": "Attention count in the 7 days before starting a new position",
        "meaning": "In the 7 days before this purchase, how many times did the user view this instrument?",
        "how": "Count behavioral events by instrument within the 7 calendar days before the purchase date.",
        "unit": "Times",
    },
    "pre_entry_attention_30d": {
        "label": "Attention count in the 30 days before starting a new position",
        "meaning": "In the 30 days before this purchase, how many times did the user view this instrument?",
        "how": "Count behavioral events by instrument within the 30 calendar days before the purchase date.",
        "unit": "Times",
    },
    "attention_days": {
        "label": "Number of attention days",
        "meaning": "The number of distinct days on which this instrument was monitored during the observation window.",
        "how": "Count distinct dates of behavioral events by instrument.",
        "unit": "Days",
    },
    "days_held": {
        "label": "Number of days held",
        "meaning": "The number of distinct days on which this instrument was held during the observation window.",
        "how": "Count by the number of days each instrument appears in holding snapshots.",
        "unit": "Days",
    },
    "attention_persistence": {
        "label": "Attention persistence",
        "meaning": "Whether attention to the instrument is sustained over the long term rather than occasional. 0–1, higher means more sustained.",
        "how": "The number of days with behavioral events for the instrument ÷ observation window days.",
        "unit": None,
    },
    "holding_persistence": {
        "label": "Holding persistence",
        "meaning": "The proportion of time the instrument was held. 0–1, higher means a more long-term core holding.",
        "how": "The number of days the instrument was held ÷ observation window days.",
        "unit": None,
    },
    "attention_capital_gap": {
        "label": "Difference between attention and allocation",
        "meaning": "Attention share − holding share. Positive = watched more but bought less (candidate pool); negative = bought more but watched less (passive or residual positions).",
        "how": "The instrument's attention share minus its position weight.",
        "unit": "%",
    },
    "attention_share": {
        "label": "Attention share",
        "meaning": "The proportion of attention received by the instrument out of attention to all instruments.",
        "how": "The number of events for the instrument ÷ the number of events for all instruments.",
        "unit": "%",
    },
    "position_weight": {
        "label": "Position proportion",
        "meaning": "The instrument's market value as a proportion of the securities' market value.",
        "how": "Market value of this instrument ÷ total securities market value.",
        "unit": "%",
    },

    # ==== 下列条目补齐前端已在展示、但此前没有释义的指标 ====================
    # 加这些不是补充说明，而是修正一个真实缺陷：页面早就在显示这些数字，却从未
    # 告诉读者它们是什么。`tests/test_metric_glossary.py` 会扫描页面里每一个
    # box("标签")，因此这里漏一个就会让测试失败。
    "activity.activity_rate": {
        "label": "Activity rate",
        "meaning": "Proportion of days with executed trades during the observation window, measuring whether trading occurs continuously.",
        "how": "Number of days with executions ÷ number of trading days in the observation window.",
        "unit": "%",
    },
    "activity.max_orders_per_day": {
        "label": "Maximum per day",
        "meaning": "Maximum number of trades on a single trading day during the observation window, reflecting the pace at peak activity.",
        "how": "Count the number of trades by trading day and take the maximum.",
        "unit": "transactions",
    },
    "activity.unique_tickers": {
        "label": "Number of instruments",
        "meaning": "Number of distinct instruments involved in trades during the observation window, reflecting trading coverage.",
        "how": "Count distinct instruments.",
        "unit": "Only",
    },
    "activity.hhi_index": {
        "label": "Concentration HHI",
        "meaning": "Whether trade value is concentrated in a small number of instruments. The closer to 1, the more concentrated.",
        "how": "Sum of squared transaction amount shares across instruments.",
        "unit": None,
        "limit": "This is concentration on a trading basis, not the same measure as concentration on a position basis (the \"position concentration\" in the attention/funds cross-tab).",
    },
    "execution.median_hold_days": {
        "label": "Median holding days",
        "meaning": "Median of matched round trips sorted by holding days; less affected by extreme values than the average.",
        "how": "Median holding days across rounds.",
        "unit": "Days",
    },
    "execution.winner_hold_days": {
        "label": "Holding of profitable positions",
        "meaning": "Average number of days profitable rounds were held.",
        "how": "Average holding days for profitable rounds.",
        "unit": "Days",
        "limit": "Only meaningful when compared with \"loss-making position holdings\": generally \"short gains, long losses\" is the typical shape of the disposition effect.",
    },
    "execution.loser_hold_days": {
        "label": "Losing-trade positions",
        "meaning": "Average number of days held in losing rounds.",
        "how": "Average holding days for losing rounds.",
        "unit": "Days",
    },
    "results.avg_win": {
        "label": "Average profit",
        "meaning": "Average profit amount per profitable round.",
        "how": "Total profit and loss of profitable rounds ÷ number of profitable rounds.",
        "unit": "Yuan",
    },
    "results.avg_loss": {
        "label": "Average loss",
        "meaning": "Average loss amount per losing round (negative value).",
        "how": "Total profit/loss of losing rounds ÷ number of losing rounds.",
        "unit": "Yuan",
        "limit": "With few samples, a single large loss or gain can significantly change this number.",
    },
    "results.risk_reward_ratio": {
        "label": "Risk-reward ratio",
        "meaning": "Average profit ÷ absolute value of average loss, measuring how many losses one win can offset.",
        "how": "|average profit| ÷ |average loss|.",
        "unit": None,
        "limit": "Considers single-instance strength only; win rate is excluded. A low win rate with a high return-to-risk ratio may still be profitable overall.",
    },
    "results.pnl_skewness": {
        "label": "Profit and loss skewness",
        "meaning": "Skew direction of the profit and loss distribution: positive indicates that a small number of large gains raised the average, while negative indicates that a small number of large losses lowered the average.",
        "how": "Skewness of profit and loss from matched round trips (third standardized moment).",
        "unit": None,
        "limit": "Sufficiently many rounds are required for statistical significance; this value is unstable with small samples.",
    },
    "results.pnl_kurtosis": {
        "label": "Kurtosis",
        "meaning": "Whether the profit and loss distribution is more concentrated in the middle than a normal distribution, with more extreme values. Higher values indicate more extreme outcomes.",
        "how": "Kurtosis of profit and loss from matched round trips (fourth standardized moment).",
        "unit": None,
        "limit": "Same as skewness; unstable when the sample is insufficient.",
    },
    "behavior.avg_down_hold_ratio": {
        "label": "Loss resistance ratio",
        "meaning": "Multiple of the holding time for unrealized-loss positions relative to unrealized-gain positions, measuring whether there is a greater willingness to \"hold through losses\".",
        "how": "Average holding days per unrealized-loss round ÷ average holding days per unrealized-gain round.",
        "unit": "Times",
        "limit": "Only a time comparison; it cannot explain why the position was held through the loss. It must be assessed together with subsequent add/trim behavior.",
    },
    "behavior.quick_profit_taking_ratio": {
        "label": "Short-term take-profit ratio",
        "meaning": "Share of profitable rounds with very short holding periods, measuring whether there is a tendency to \"take a small profit and exit\".",
        "how": "Short-term profitable rounds ÷ profitable rounds.",
        "unit": "%",
    },
    "capital_management.avg_invested_ratio": {
        "label": "Average position share",
        "meaning": "Average securities-market-value-to-total-assets ratio during the observation window, reflecting how much money is normally in the market.",
        "how": "Mean of daily (securities market value ÷ total assets).",
        "unit": "%",
        "limit": "Counts only trading days with valid asset composition; dates with abnormal deposits or withdrawals have been excluded.",
    },
    "capital_management.cash_buffer_ratio": {
        "label": "Average cash share",
        "meaning": "Average cash-to-total-assets ratio during the observation window, reflecting the amount of buffer retained.",
        "how": "Mean of daily (cash ÷ total assets).",
        "unit": "%",
        "limit": "Complements \"average position share\"; the two usually add up to approximately 100%.",
    },
    "capital_management.invested_ratio_range": {
        "label": "Position adjustment magnitude",
        "meaning": "The fluctuation range of position allocation during the observation window (maximum − minimum), measuring the activity level of market timing.",
        "how": "Maximum daily position ratio minus minimum daily position ratio.",
        "unit": "%",
    },
    "capital_management.risk_on_off_switch_count": {
        "label": "Risk on/off switching",
        "meaning": "The number of times the position switches back and forth between \"clearly fully invested\" and \"clearly lightly invested\".",
        "how": "Determine the daily state as offensive/defensive based on thresholds, and count state transitions.",
        "unit": "Times",
    },
    "capital_management.exposure_style": {
        "label": "Position style",
        "meaning": "Whether the overall position is fully invested over the long term or has significant fluctuations.",
        "how": "Classify by position level and magnitude of its fluctuations.",
        "unit": None,
    },
    "position_style.opens": {
        "label": "Number of position openings",
        "meaning": "Number of times positions were established from zero during the observation window.",
        "how": "Compare holdings daily; count once when a new instrument appears (instruments held at the start of the observation window are excluded).",
        "unit": "Times",
    },
    "position_style.adds": {
        "label": "Number of adds",
        "meaning": "Number of times the quantity of an existing position was increased.",
        "how": "Compare the number of holdings daily; count once when an increase occurs.",
        "unit": "Times",
    },
    "position_style.reduces": {
        "label": "Number of trims",
        "meaning": "Number of times the quantity of an existing position was reduced without being fully exited.",
        "how": "Compare the number of holdings daily; count once when a decrease occurs and the position is still held.",
        "unit": "Times",
    },
    "position_style.closes": {
        "label": "Number of Exits",
        "meaning": "Number of times an instrument's position was reduced to zero.",
        "how": "Compare holdings daily; count once when an instrument changes from present to absent.",
        "unit": "Times",
    },
    "position_style.entry_style": {
        "label": "Position-opening method",
        "meaning": "Whether positions are usually opened in a single purchase or built over several purchases.",
        "how": "Classify by whether buying continued on the same or an adjacent trading day after starting a new position.",
        "unit": None,
    },
    "position_style.exit_style": {
        "label": "Exit Method",
        "meaning": "Whether exits are typically completed in one sale or gradually through several transactions.",
        "how": "Classify by the number of position trims before exiting.",
        "unit": None,
    },
    "position_style.average_down_rate": {
        "label": "Unrealized-loss add-on share",
        "meaning": "Proportion of adding behavior occurring while positions have unrealized losses.",
        "how": "Number of add-ons under unrealized losses ÷ total number of add-ons.",
        "unit": "%",
        "limit": "Unrealized-loss add-ons may reflect active averaging down or completing the originally planned position; this metric does not distinguish the motivation.",
    },
    "position_style.pyramid_winner_rate": {
        "label": "Unrealized-Profit Add-Position Share",
        "meaning": "Proportion of adding behavior occurring while positions have unrealized gains.",
        "how": "Number of Additions During Unrealized Profit ÷ Total Number of Additions.",
        "unit": "%",
    },
    "position_style.loser_reduce_rate": {
        "label": "Trim targeting losses",
        "meaning": "Proportion of trim actions targeting positions with floating losses. Higher = more inclined to cut losing positions.",
        "how": "Number of trims under unrealized losses ÷ total number of trims.",
        "unit": "%",
    },
    "position_style.winner_reduce_rate": {
        "label": "Profit-targeted trimming",
        "meaning": "Proportion of trim actions targeting positions with floating gains. Higher = more inclined to lock in gains.",
        "how": "Number of Trims During Unrealized Profit ÷ Total Number of Trims.",
        "unit": "%",
    },
    "position_style.full_exit_ratio": {
        "label": "Proportion of full-position exits",
        "meaning": "Proportion of exit behavior involving selling the entire position at once.",
        "how": "One-time exit count ÷ total exit count.",
        "unit": "%",
    },
    "position_style.partial_exit_ratio": {
        "label": "Share of partial-position trims",
        "meaning": "Proportion of exit behavior involving selling in multiple transactions.",
        "how": "Number of times positions were trimmed before exit ÷ total number of exits.",
        "unit": "%",
    },
}

# ---------------------------------------------------------------------------
# Mapping to the four decision-funnel stages, for the causal-chain view
# ---------------------------------------------------------------------------
# The funnel counts live under `attention_capital.decision_funnel.counts` keyed by
# these names. Kept here so the glossary and the funnel chart cannot drift apart.
FUNNEL_STAGES = (
    ("attended_tickers", "Followed instruments", "Instruments with behavioral events"),
    ("reached_research", "Reached research-grade", "Behavior reaches \"research\" depth, rather than just a glance"),
    ("observable_entries", "Observable position initiation", "Establishing a position from zero"),
    ("entries_scaled_up", "Subsequent addition to a position", "Increased the position after starting a new position"),
    ("entries_held_30d_plus", "Held for at least 30 days", "Continue holding for at least 30 days after purchase"),
)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
def _fmt(d):
    return str(d)[:10] if d else None


def _span(p, days):
    """Range string for a trailing window, anchored on the newest data date."""
    dr = ((p.get("asset") or {}).get("date_range") or [None, None])
    end = _fmt(dr[1]) or _fmt((p.get("attention_capital") or {}).get("positions_as_of"))
    if not end:
        return None, None
    try:
        e = datetime.fromisoformat(end)
        return _fmt(e - timedelta(days=days - 1)), end
    except ValueError:
        return None, end


def window_for(path, p):
    """The period a metric was computed from, as {label, range, note}.

    Derived from the block that owns the metric, never hardcoded: the real window
    travels with the data, and several blocks deliberately use different ones.
    """
    p = p or {}
    ac = p.get("attention_capital") or {}
    pm = p.get("portfolio_metrics") or {}
    pd = p.get("position_dynamics") or {}
    tm = p.get("trade_metrics") or {}
    asset = p.get("asset") or {}
    fc = p.get("action_forecast") or {}

    # 关注/资金交叉：整个块共用一个窗口
    if path.startswith("attention_capital") or path in (
            "loss_attention_ratio", "attention_on_losing_positions",
            "attention_on_winning_positions", "share_on_losing",
            "attention_share", "position_weight", "attention_capital_gap",
            "attention_hhi", "portfolio_hhi", "held_ticker_view_ratio",
            "nonheld_ticker_view_ratio", "prior_attention_rate_7d",
            "research_to_holding_conversion", "attention_days", "days_held",
            "attention_persistence", "holding_persistence",
            "attended_tickers", "reached_research", "observable_entries",
            "entries_scaled_up", "entries_held_30d_plus",
            "attention_to_research", "research_to_entry", "entry_to_scale_up",
            "entry_to_persistent_holding"):
        w = ac.get("window") or [None, None]
        d = ac.get("window_days")
        if path in ("prior_attention_rate_7d", "pre_entry_attention_7d"):
            note = (f'"7 days before starting a position" is calculated separately as the 7 calendar days before each purchase; the overall observation window is {d} days.')
            return {"label": "7 days before starting a new position (per event)", "range": f"{_fmt(w[0])} → {_fmt(w[1])}",
                    "note": note}
        if path == "pre_entry_attention_30d":
            return {"label": "30 days before starting a new position (per event)",
                    "range": f"{_fmt(w[0])} → {_fmt(w[1])}", "note": None}
        return {"label": f"Past {d} days" if d else None,
                "range": f"{_fmt(w[0])} → {_fmt(w[1])}" if w[0] else None,
                "note": ac.get("baseline_note") if path != "research_to_holding_conversion" else None}

    # 持仓动态
    if path.startswith("position_dynamics") or path in (
            "rotation.composition_change_rate", "avg_similarity_day_to_day",
            "conviction.max_position_weight_ever", "reentry_rate",
            "usable_ratio", "carried_forward_days",
            "average_down_tendency", "pyramid_winner_tendency"):
        d = pd.get("window_days")
        return {"label": f"Past {d} days" if d else None, "range": None,
                "note": "Position comparison is based on daily position snapshots; the previous day's position is carried forward on days with missing data."}

    # 下一步动作推演：与持仓动态同一窗口（习惯是按这个窗口观测到的）
    if path.startswith("action_forecast"):
        d = pd.get("window_days") or (fc.get("window_days"))
        w = fc.get("window") or pd.get("window") or [None, None]
        return {"label": f"Past {d} days (habit observation window)" if d else None,
                "range": f"{_fmt(w[0])} → {_fmt(w[1])}" if w and w[0] else None,
                "note": "The prediction window is the usual observation window; once user behavior deviates from the pattern within this window, the prediction becomes invalid."}

    # 成交
    if path.startswith("activity.") or path.startswith("execution.") or \
            path.startswith("results.") or path == "activity.total_orders":
        d = (tm.get("activity") or {}).get("observation_days")
        return {"label": f"Past {d} days" if d else None,
                "range": None,
                "note": "Count by actual execution time (not the trade_date field)."}

    # 组合指标：短/中/长期口径不同，必须分别标注
    if path.startswith("short_term_trading.") or path in (
            "portfolio_snapshot.total_market_value", "portfolio_snapshot.num_positions",
            "weekly_pnl", "weekly_return_pct"):
        d = ((pm.get("short_term") or {}).get("short_term_trading") or {}).get("window_days") or 7
        s, e = _span(p, d)
        return {"label": f"Past {d} days", "range": f"{s} → {e}" if s else None,
                "note": "The position snapshot uses the most recent trading day with holdings."}
    if path.startswith("market_value_trend.") or path.startswith("pnl_trend.") or \
            path.startswith("position_habit.") or path == "medium_term_trading.monthly_turnover":
        d = (pm.get("medium_term") or {}).get("window_days") or 28
        s, e = _span(p, d)
        return {"label": f"Past {d} days", "range": f"{s} → {e}" if s else None, "note": None}

    # 长期 / 净值曲线家族
    if path.startswith("asset.") or path.startswith("risk_exposure.") or \
            path.startswith("holding_endurance.") or path.startswith("performance."):
        d = asset.get("num_days") or ((pm.get("long_term") or {}).get("performance") or {}).get("observation_days")
        dr = asset.get("date_range") or [None, None]
        return {"label": f"Entire observation window ({d} days)" if d else "Full observation window",
                "range": f"{_fmt(dr[0])} → {_fmt(dr[1])}" if dr[0] else None,
                "note": "Returns/drawdown/volatility belong to the same equity curve, with a single definition."}

    # 其余：退回组合指标的观察期
    d = pm.get("observation_days") or asset.get("num_days")
    s, e = _span(p, d) if d else (None, None)
    return {"label": f"Observation window ({d} days)" if d else None,
            "range": f"{s} → {e}" if s else None, "note": None}


def describe(path):
    """Glossary entry for a metric path, or None when it has no explanation yet."""
    return ENTRIES.get(path)


def annotations(profile, paths=None):
    """`{path: {label, meaning, how, unit, limit, window}}` for the viewer.

    Derived on read — nothing is stored — so editing the glossary reaches every
    stored profile immediately.
    """
    p = profile or {}
    out, missing = {}, []
    for path in (paths if paths is not None else ENTRIES):
        e = describe(path)
        if not e:
            missing.append(path)
            continue
        out[path] = {**e, "window": window_for(path, p)}
    return out, missing
