"""Indirect (proxy) KYC features.

The principle: when a KYC feature needs user-provided information we do not
have, do NOT invent it — but also do not stop at "unknown" when the data we *do*
have supports a defensible indirect estimate. An advisor is better served by
"账户内可用现金约可覆盖 3.4 个月的净转出（代理口径，未含外部支出）" than by a
blank field.

Every value from this module is marked:

    is_proxy = True
    proxy_basis  — what it is derived from, so the inference is auditable
    caveat       — what it is NOT, so it can never be mistaken for the real fact
    confidence   — capped low, because an inference is not an observation

Two kinds of proxy live here:

  * **Bounded statements** — a fact that is true *within the observed scope*,
    e.g. the platform only holds listed securities, so no illiquid asset is
    visible. True, useful, and explicitly scoped.
  * **Derived estimates** — a measured quantity recomputed with a different
    denominator, e.g. cash ÷ observed net withdrawal rate = a runway in months.
"""
import statistics
from collections import Counter, defaultdict

from .account import (adjusted_equity_series, external_flows, max_drawdown)
from .kyc_features import _f, _latest_book
from .taxonomy import country_of, ticker_info

PROXY_CONF_CAP = 0.6          # an inference must never look like an observation


def _p(value, display, basis, caveat, evidence, conf, estimate=True):
    """Build a proxy leaf.

    `estimate=False` marks a *bounded scope statement* — "within what we can see,
    nothing was found" — rather than a derived quantity. Those still carry real
    information (an advisor learns we looked), but they do not answer the KYC
    question, so the tree must not count them as covered.
    """
    return {
        "value": value, "display": display,
        "evidence": evidence, "confidence": min(conf, PROXY_CONF_CAP),
        "is_proxy": True, "estimate": estimate,
        "proxy_basis": basis, "caveat": caveat,
    }


# ---------- portfolio exposure ----------

def geographic_and_currency_exposure(acct, trades=None, intent=None):
    """Non-US exposure via issuer domicile of the held instruments."""
    as_of, book = _latest_book(acct)
    if not book:
        return None
    by_country = defaultdict(float)
    for code, p in book.items():
        by_country[country_of(code)] += _f(p.get("stock_market_value"), 0.0) or 0.0
    total = sum(by_country.values()) or 0.0
    if total <= 0:
        return None
    shares = {k: round(v / total, 4) for k, v in sorted(by_country.items(), key=lambda kv: -kv[1])}
    non_us = 1 - shares.get("United States", 0.0)
    foreign = [k for k in shares if k != "United States"]
    display = (f"Positions are primarily in U.S.-listed securities ({shares.get('United States', 0) * 100:.0f}%); non-U.S. exposure is approximately {non_us * 100:.0f}%" +
               (f", primarily from {'、'.join(foreign[:3])}" if foreign else ""))
    return _p(
        {"as_of": as_of, "by_country": shares, "non_us_share": round(non_us, 4)},
        display,
        "Classify by the issuer's registered location for held instruments (US-listed ADRs/foreign ordinary shares are counted as non-US based on registered location)",
        "Only covers the instrument's place of registration, not the currency or actual trading currency; instruments not listed are treated as U.S. stocks, so non-U.S. exposure is a lower-bound estimate",
        [f"bm453_daily[{as_of}].holdings mapped by issuer domicile",
         f"Non-U.S. instruments {'、'.join(str(x) for x in foreign[:5])} collectively account for {non_us * 100:.1f}% of securities market value"],
        0.55)


def illiquid_asset_exposure(acct, trades=None, intent=None):
    """Bounded statement: this platform's book contains listed securities only."""
    as_of, book = _latest_book(acct)
    if not book:
        return None
    classes = Counter(ticker_info(c)["sector"] for c in book)
    return _p(
        {"as_of": as_of, "positions": len(book), "sectors": dict(classes),
         "illiquid_visible": False},
        f"A total of {len(book)} positions in platform-visible accounts, all securities tradable on the secondary market; no illiquid assets observed (private funds, unlisted equity, real estate, etc. are outside the visible scope)",
        "Tradability of account holding details",
        "This is a qualified conclusion of \"zero within the visible scope\" and does not mean the user has no illiquid assets outside their platform",
        [f"bm453_daily[{as_of}].holdings are all exchange-listed instruments"],
        0.5, estimate=False)


def external_asset_coverage(acct, trades=None, intent=None):
    """Bounded statement: we only see this platform's account."""
    if not acct:
        return None
    return _p(
        {"visible_scope": "platform_account_only"},
        "This profile covers only the user's securities account on the platform; external assets such as bank accounts, other brokerage accounts, retirement accounts, and real estate are not visible, so any conclusion about total assets is incomplete",
        "Data access scope (BM453 is the platform account snapshot)",
        "This is not an assessment of the user's total assets, only a description of the visible scope",
        ["BM453 account snapshots contain only platform accounts"],
        0.6, estimate=False)


def employer_stock_concentration(acct, trades=None, intent=None):
    """Employer identity is unknown, so report the risk-equivalent signal:
    a very concentrated single-name position, which is the exposure an employer
    -stock problem would create."""
    as_of, book = _latest_book(acct)
    if not book:
        return None
    total = sum(_f(p.get("stock_market_value"), 0.0) or 0.0 for p in book.values())
    if total <= 0:
        return None
    code, top_mv = max(
        ((c, _f(pos.get("stock_market_value"), 0.0) or 0.0) for c, pos in book.items()),
        key=lambda kv: kv[1])
    share = top_mv / total
    if share < 0.3 or len(book) < 2:
        return _p(
            {"as_of": as_of, "top1_share": round(share, 4), "flag": False},
            f"No excessive concentration in a single holding observed (maximum holding {code} accounts for {share * 100:.0f}%), so no employer-stock-like concentrated risk pattern exists",
            "Maximum single-position share of securities market value",
            "Cannot identify whether the holding is employer stock (no employer information); only the concentration pattern is assessed here",
            [f"bm453_daily[{as_of}].holdings top1={code} {share * 100:.1f}%"],
            0.5)
    return _p(
        {"as_of": as_of, "top1_share": round(share, 4), "top1": code, "flag": True},
        f"A single holding is highly concentrated: {code} accounts for {share * 100:.0f}% of securities market value. Cannot confirm whether it is employer stock (employer information unavailable), but its risk profile is consistent with employer stock concentration",
        "Maximum single-position share (employer identity unavailable, replaced with equivalent risk profile)",
        "Cannot confirm the relationship between this instrument and the user's occupation; this describes concentration risk, not an employer stock conclusion",
        [f"bm453_daily[{as_of}].holdings top1={code} {share * 100:.1f}%",
         f"Total {len(book)} positions, total securities market value {total:,.0f}"],
        0.5)


def account_or_goal_specific_risk_level(acct, trades=None, intent=None):
    """Scope statement: which account/goal this risk assessment applies to."""
    if not acct:
        return None
    return _p(
        {"scope": "single_platform_account"},
        "This risk assessment applies only to the platform account as a whole and is not split by goal or across multiple accounts; if the user has other accounts or funds for different purposes, the risk level must be assessed separately",
        "Data access scope (single-account snapshot)",
        "This is a scope description, not risk levels assigned separately to each account",
        ["BM453 single-account basis"],
        0.6, estimate=False)


# ---------- financial capacity ----------

def _monthly_outflow_rate(acct, trades):
    """Average net withdrawal per month over the observed window."""
    flows = external_flows(acct, trades)
    if not flows:
        return None, None, None
    dates = sorted({f["date"] for f in flows})
    net_out = sum(-f["amount"] for f in flows if f["amount"] < 0)
    net_in = sum(f["amount"] for f in flows if f["amount"] > 0)
    span = len(acct)
    months = span / 30.0
    return (net_out / months if months else None, net_in / months if months else None, flows)


def emergency_fund_runway(acct, trades=None, intent=None):
    """Cash balance ÷ observed monthly net outflow, in months."""
    if not acct:
        return None
    cash = _f(acct[-1].get("money_remain"))
    if cash is None:
        return None
    out_pm, in_pm, flows = _monthly_outflow_rate(acct, trades or [])
    if not flows:
        return _p(
            {"cash": round(cash, 2), "months": None, "observed_flows": 0},
            f"Account cash is approximately {cash:,.0f}; no identifiable net outflows occurred during the observation window, so the cash-support duration cannot be estimated from this basis (this does not mean there were no external expenses)",
            "Cash balance and observed outflow records",
            "Only counts transfers out within the platform; the user's daily spending is not visible, so this conclusion cannot be equated with emergency fund coverage",
            [f"bm453_daily.money_remain={cash}", "No external transfer-out records"],
            0.45, estimate=False)
    if not out_pm:
        return _p(
            {"cash": round(cash, 2), "months": None, "observed_flows": len(flows)},
            f"Account cash is approximately {cash:,.0f}; all cash flows during the observation window were net inflows, with no outflow baseline available for estimating the support duration",
            "Cash balance and net inflow records",
            "Supportable months can only be estimated after net outflows occur",
            [f"bm453_daily.money_remain={cash}", f"All {len(flows)} deposits and withdrawals are net inflows"],
            0.45, estimate=False)
    months = cash / out_pm
    # A runway is only meaningful against a real spending baseline. A handful of
    # tiny transfers (e.g. $12/month) would otherwise produce "covers 48 months",
    # which is arithmetically true and analytically worthless.
    thin = len([f for f in flows if f["amount"] < 0]) < 2 or out_pm < 0.01 * max(cash, 1)
    if thin:
        return _p(
            {"cash": round(cash, 2), "monthly_net_outflow": round(out_pm, 2),
             "months": None, "observed_flows": len(flows)},
            f"Account cash is approximately {cash:,.0f}; however, during the observation window there were only {len([f for f in flows if f['amount'] < 0])} outgoing transfers, averaging {out_pm:,.0f} per month, with amounts too small to serve as a spending baseline, so no supportable number of months is provided (to avoid producing a meaningless long-horizon estimate)",
            "Whether cash balance and observed outflows constitute a usable baseline",
            "Too few transfer-out samples or amounts too small; actual emergency coverage requires expenditure information, and in-platform data cannot provide a reliable value",
            [f"bm453_daily.money_remain={cash}",
             f"Monthly average outflows {out_pm:,.0f}, accounting for {out_pm / max(cash, 1) * 100:.2f}% of cash"],
            0.35, estimate=False)
    return _p(
        {"cash": round(cash, 2), "monthly_net_outflow": round(out_pm, 2),
         "months": round(months, 1), "observed_flows": len(flows)},
        f"Based on the average monthly net outflow of {out_pm:,.0f} during the observation window, cash {cash:,.0f} in the account can cover approximately {months:.1f} months; this is the lower bound on an account basis and excludes the user's external cash and income",
        "Cash balance ÷ observed average monthly net outflows (BM453 cash flow inference)",
        "Covers only transfers out within the platform, excluding daily consumption and spending from external accounts; the sample period is limited, so average monthly outflows may not represent long-term levels",
        [f"bm453_daily.money_remain={cash}",
         f"Inferred average monthly net outflow {out_pm:,.0f} ({len(flows)} transactions)"],
        0.4)


def total_debt_band(acct, trades=None, intent=None):
    """In-account debt detected as stock value exceeding net assets (margin)."""
    if not acct:
        return None
    days = [d for d in acct
            if _f(d.get("stock_total")) is not None and _f(d.get("asset_total")) is not None
            and _f(d.get("stock_total")) > _f(d.get("asset_total")) * 1.01]
    max_gap, worst_date = 0.0, None
    for d in days:
        gap = (_f(d.get("stock_total")) - _f(d.get("asset_total")))
        if gap > max_gap:
            max_gap, worst_date = gap, d.get("snapshot_date")
    if not days:
        return _p(
            {"in_account_debt_detected": False, "days_with_margin": 0},
            "No instance during the observation window where securities market value exceeded total assets; no margin liabilities detected in the account",
            "Relationship between stock_total and asset_total in BM453",
            "Can only reflect financing within this platform account; external liabilities such as mortgages and consumer loans are entirely unobservable.",
            [f"bm453_daily stock_total ≤ asset_total on {len(acct)} days"],
            0.55)
    return _p(
        {"in_account_debt_detected": True, "days_with_margin": len(days),
         "peak_margin_amount": round(max_gap, 2), "peak_date": worst_date},
        f"Detected account financing: securities market value exceeded total assets on {len(days)} days, with a peak of approximately {max_gap:,.0f} ({worst_date})",
        "Number of days and difference where stock_total > asset_total in BM453",
        "Only covers margin liabilities within the account and excludes various loans outside the account; the difference may also be affected by the valuation time",
        [f"In bm453_daily, stock_total > asset_total×1.01 on {len(days)} days",
         f"Peak gap {max_gap:,.0f} ({worst_date})"],
        0.5)


def _contribution_cadence(acct, trades):
    flows = external_flows(acct, trades)
    ins = [f for f in flows if f["amount"] > 0]
    if not ins:
        return None
    amounts = [f["amount"] for f in ins]
    span_months = max(len(acct) / 30.0, 1)
    return {"count": len(ins), "amounts": amounts,
            "monthly": round(sum(amounts) / span_months, 2),
            "median": round(statistics.median(amounts), 2),
            "cv": round(statistics.pstdev(amounts) / (sum(amounts) / len(amounts)), 3)
            if len(amounts) > 1 and sum(amounts) else None,
            "dates": [f["date"] for f in ins]}


def planned_contribution(acct, trades=None, intent=None):
    """Observed contribution pattern as a proxy for the stated funding plan."""
    if not acct:
        return None
    cad = _contribution_cadence(acct, trades or [])
    if not cad:
        return _p(
            {"observed": False, "events": 0},
            "No external deposits identified during the observation window, so no funding plan can be inferred; funds may have already been in place, or the deposit may have occurred outside this platform",
            "BM453 inferred deposits and withdrawals",
            "No deposit records do not mean the user has no investment plan; they only indicate that no deposits occurred within the observable window",
            ["No external deposit records"],
            0.4, estimate=False)
    regular = cad["cv"] is not None and cad["cv"] <= 0.5
    return _p(
        {"observed": True, "events": cad["count"], "monthly_avg": cad["monthly"],
         "median_amount": cad["median"], "regularity_cv": cad["cv"],
         "regular": regular},
        (f"{cad['count']} deposits occurred during the observation window, equivalent to approximately {cad['monthly']:,.0f} per month,"
         + ("Amount is fairly regular, close to a fixed investment rhythm" if regular else
            "Amount fluctuates significantly, more like opportunistic investment than fixed regular investment")),
        "Number, amount, and time distribution of deposits inferred by BM453",
        "This is deposited-funds behavior that has already occurred, not the user's declared future investment plan; plans outside the window are not visible",
        [f"Inferred deposits: {cad['count']} transactions, median amount {cad['median']:,.0f}",
         f"Dates {'、'.join(str(d) for d in cad['dates'][:4])}"],
        0.45)


def planned_withdrawal(acct, trades=None, intent=None):
    """Observed withdrawal pattern as a proxy for the stated withdrawal plan."""
    if not acct:
        return None
    flows = external_flows(acct, trades or [])
    outs = [f for f in flows if f["amount"] < 0]
    if not outs:
        return _p(
            {"observed": False, "events": 0},
            "No external withdrawals identified during the observation window, so no withdrawal plan can be inferred; this may mean funds remain in an ongoing investment phase, or the withdrawal occurred through another account",
            "BM453 inferred deposits and withdrawals",
            "No withdrawal records do not mean there is no withdrawal demand; they only indicate that no withdrawals occurred within the observable window",
            ["No external withdrawal records"],
            0.4, estimate=False)
    span_months = max(len(acct) / 30.0, 1)
    monthly = sum(-f["amount"] for f in outs) / span_months
    return _p(
        {"observed": True, "events": len(outs), "monthly_avg": round(monthly, 2),
         "dates": [f["date"] for f in outs]},
        f"{len(outs)} withdrawals occurred during the observation window, equivalent to approximately {monthly:,.0f} per month; the account has a need for fund withdrawals, so return evaluation requires assessment by time and permissions",
        "Number and amount of withdrawals inferred by BM453",
        "Withdrawals already made do not equal planned withdrawals; purpose unknown",
        [f"Inferred {len(outs)} withdrawals, monthly average {monthly:,.0f}",
         f"Dates {'、'.join(str(f['date']) for f in outs[:4])}"],
        0.45)


def income_stability(acct, trades=None, intent=None):
    """Contribution regularity as a *weak* proxy for income stability."""
    if not acct:
        return None
    cad = _contribution_cadence(acct, trades or [])
    if not cad or cad["count"] < 3:
        return _p(
            {"basis": "insufficient_contributions", "events": (cad or {}).get("count", 0)},
            "Fewer than 3 funding inflow records; income stability cannot be indirectly observed from the regularity of funding supply",
            "Funding inflow count",
            "Income stability requires career and income information; regularity of fund supply is only a very weak proxy signal",
            [f"Deposits {((cad or {}).get('count') or 0)} times"],
            0.3, estimate=False)
    cv = cad["cv"]
    label = "Capital supply is relatively regular" if cv is not None and cv <= 0.5 else "Capital supply is highly volatile"
    return _p(
        {"basis": "contribution_regularity", "events": cad["count"],
         "regularity_cv": cv, "monthly_avg": cad["monthly"]},
        f"{label}: dispersion of the amounts of {cad['count']} deposits {cv}, monthly average approximately {cad['monthly']:,.0f}",
        "Coefficient of variation of funding inflow amounts (proxy for income/savings capacity stability)",
        "This is the regularity of funding supply, not proof of income; income may come from non-recurring sources, and external funds may also exist",
        [f"Deposits {cad['count']} transactions, coefficient of variation {cv}", f"Monthly average {cad['monthly']:,.0f}"],
        0.35)


def annual_income_band(acct, trades=None, intent=None):
    """Annualised observed contribution as a *lower-bound* savings-capacity proxy."""
    if not acct:
        return None
    cad = _contribution_cadence(acct, trades or [])
    if not cad or cad["monthly"] <= 0:
        return _p(
            {"basis": "no_contributions"},
            "No deposit records during the observation window that can be used to infer savings capacity; income level requires user input",
            "Funding inflow records",
            "No deposits does not mean no income; it only indicates that no funds were transferred in during the observable window",
            ["No deposit records"],
            0.25, estimate=False)
    annual = cad["monthly"] * 12
    band = ("<10,000" if annual < 10_000 else "10,000-50,000" if annual < 50_000 else
            "50,000-200,000" if annual < 200_000 else "200,000-1,000,000" if annual < 1_000_000 else ">1 million")
    return _p(
        {"annualized_contribution": round(annual, 2), "band": band,
         "events": cad["count"]},
        f"Annualized based on deposit cadence during the observation window: approximately {annual:,.0f}, falling in the {band} range; this is only the **lower bound of discretionary surplus**, not the same concept as income level",
        "Monthly average deposits × 12 (lower-bound proxy for savings capacity)",
        "Funding inflows only reflect disposable surplus transferred into this platform, far below actual income; must not be used as an income conclusion",
        [f"Monthly average deposits {cad['monthly']:,.0f} → annualized {annual:,.0f}"],
        0.25)


def maximum_affordable_loss(acct, trades=None, intent=None):
    """Observed peak-to-trough as a *lower bound* on affordable loss.

    A user who lived through a -30% drawdown without liquidating has demonstrably
    tolerated that much. This is a floor established by behaviour, not a stated
    limit — the questionnaire figure could be higher.
    """
    if not acct:
        return None
    adj = adjusted_equity_series(acct, trades)
    dates, values = adj["dates"], adj["values"]
    if len(values) < 2 or not adj.get("reliable", True):
        return _p(
            {"basis": "insufficient_curve"},
            "Insufficient account curve coverage to estimate the lower bound of tolerable loss based on drawdowns actually experienced",
            "Time-weighted equity series",
            "Can only describe drawdown visible in platform accounts; it does not represent the larger losses the user is willing or able to bear",
            ["Fewer than 50% of trading days available for the curve"],
            0.3, estimate=False)
    dd, dd_date, peak_at, _rec = max_drawdown(dates, values)
    if not dd:
        return _p(
            {"tolerated_drawdown_pct": 0.0, "observed_peak": peak_at},
            "No significant account drawdown formed during the observation window; therefore, the lower bound of tolerable losses cannot be inferred from this.",
            "Maximum drawdown of the time-weighted equity series",
            "No drawdown does not mean low risk tolerance; it only indicates that it was not tested during the observable window",
            [f"Net asset value series: {len(values)} days, maximum drawdown {dd}"],
            0.3, estimate=False)
    amount = None
    for d, v in zip(dates, values):
        if d == dd_date:
            amount = v
            break
    return _p(
        {"tolerated_drawdown_pct": round(dd * 100, 2), "drawdown_date": dd_date,
         "peak_value": peak_at, "trough_value": amount},
        f"The user actually endured approximately {abs(dd) * 100:.1f}% account drawdown (reached on {dd_date}) without exiting the market, constituting the **lower bound** of tolerable loss, not the upper bound stated by the user",
        "Maximum drawdown of the time-weighted equity curve (deposits and withdrawals excluded)",
        "This is drawdown passively experienced, not risk the user was actively willing to bear; if the user has other assets or leverage, their actual capacity to bear risk may differ",
        [f"Maximum drawdown {dd * 100:.1f}% ({dd_date})", f"Peak {peak_at:,.0f}"],
        0.45)


# ---------- registry ----------

PROXY = {
    "geographic_and_currency_exposure": geographic_and_currency_exposure,
    "illiquid_asset_exposure": illiquid_asset_exposure,
    "external_asset_coverage": external_asset_coverage,
    "employer_stock_concentration": employer_stock_concentration,
    "account_or_goal_specific_risk_level": account_or_goal_specific_risk_level,
    "maximum_affordable_loss": maximum_affordable_loss,
    "emergency_fund_runway": emergency_fund_runway,
    "total_debt_band": total_debt_band,
    "planned_contribution": planned_contribution,
    "planned_withdrawal": planned_withdrawal,
    "income_stability": income_stability,
    "annual_income_band": annual_income_band,
}


def build_proxy_features(snap, intent=None):
    acct = snap.get("account_daily") or []
    trades = snap.get("trades_6m") or []
    out = {}
    for key, fn in PROXY.items():
        error = None
        try:
            r = fn(acct, trades, intent)
        except Exception as e:                                        # noqa: BLE001
            r = None
            error = f"{type(e).__name__}: {e}"
        if r is None:
            # A crash and a genuine data gap must NOT look the same. Without
            # this an import bug or typo silently reads as "insufficient data",
            # which is exactly how a broken proxy could hide for a whole batch.
            r = {"value": None,
                 "display": ("Indirect inference calculation failed:" + error) if error
                            else "Current data is insufficient for indirect inference",
                 "evidence": [], "confidence": 0.0, "is_proxy": True,
                 "proxy_basis": "No usable input", "caveat": "Insufficient minimum data required for calculation",
                 "estimate": False}
            if error:
                r["error"] = error
        out[key] = {**r, "coverage": "proxy", "method": "rule"}
    from .english_output import englishize
    return englishize(out)
