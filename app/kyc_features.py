"""Deterministic KYC feature extraction.

Fills the `deterministic` features of app/kyc_catalog.py straight from the BM
streams: no LLM, every number reproducible, every value carries evidence refs.

Design notes worth keeping:

* Cash vs stock split comes from BM453 (`asset_total` - `stock_total`), so
  `money_remain` is checked against it rather than trusted blindly.
* Leveraged products are detected from the *instrument name* that BM453 already
  carries ("T-Rex 2X Long Alphabet Daily Target ETF"), not only from a static
  ticker list — that catches names the list would miss and is auditable.
* External cash flow is estimated only on days with no trade and no change to
  the holding set. On such a day `Δasset_total − Δstock_total` isolates money
  moving in or out, because price moves hit both sides equally. Trade days are
  excluded so buys/sells (internal transfers) cannot masquerade as deposits.
"""
import re
import statistics
from collections import Counter, defaultdict

from .account import (adjusted_equity_series, external_flows, max_drawdown,
                     total_return)
from .taxonomy import SECTOR_MAP, parse_ts, ticker_info

# ---------- instrument classification ----------

_LEV_NAME_RE = re.compile(
    r"(\b\d+(\.\d+)?x\b|\bleveraged\b|\bbull\b|\bbear\b|\binverse\b|"
    r"\blong\b|\bshort\b|\bdaily target\b|\bultra\b|\bultrapro\b)", re.I)

_BOND_HINTS = {"SGOV", "SHV", "SHY", "TLT", "IEF", "BND", "AGG", "USDY", "TFLO", "USFR", "BIL"}
_COMMODITY_HINTS = {"GLD", "SLV", "IAU", "USO", "UNG", "GDX", "XLE"}
_INDEX_ETF_HINTS = {"QQQ", "SPY", "IWM", "DIA", "VOO", "VTI", "VEA", "VWO"}


def _is_leveraged(code, name):
    """Leveraged / inverse product, by ticker list OR by instrument name."""
    if ticker_info(code)["sector"] == "Leveraged ETF":
        return True
    if code in {c for c, (s, _t) in SECTOR_MAP.items() if s == "Leveraged ETF"}:
        return True
    return bool(_LEV_NAME_RE.search(name or ""))


def classify_instrument(code, name):
    """Coarse asset class for allocation bucketing."""
    code = (code or "").upper()
    if _is_leveraged(code, name):
        return "leveraged"
    if code in _BOND_HINTS or re.search(r"treasury|bond|income|yield", name or "", re.I):
        return "fixed_income"
    if code in _COMMODITY_HINTS or re.search(r"gold|silver|oil|commodit", name or "", re.I):
        return "commodity"
    if code in _INDEX_ETF_HINTS or re.search(r"\bindex\b|\bs&p\b|nasdaq|total market", name or "", re.I):
        return "index_equity"
    if re.search(r"\betf\b|\btrust\b|\bfund\b", name or "", re.I):
        return "thematic_equity"
    return "single_name_equity"


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _band(value, edges, labels):
    if value is None:
        return None
    for e, lab in zip(edges, labels):
        if value < e:
            return lab
    return labels[-1]


def _latest_book(acct):
    """(date, holdings) of the most recent day with a non-empty book.

    A user who flattened the account before the snapshot ends would otherwise
    be reported as "100% cash", which says nothing about how they actually
    invested. The date is always surfaced so the reader knows the vintage.
    """
    for d in reversed(acct or []):
        book = d.get("holdings") or {}
        if book:
            return d.get("snapshot_date"), book
    return None, {}


# ---------- portfolio state ----------

def account_balances_and_cash(acct):
    if not acct:
        return None
    last = acct[-1]
    total = _f(last.get("asset_total"))
    stock = _f(last.get("stock_total"))
    cash = _f(last.get("money_remain"))
    if cash is None and total is not None and stock is not None:
        cash = total - stock
    if total is None and stock is not None and cash is not None:
        total = stock + cash
    if total is None:
        return None
    return {
        "value": {"total_assets": round(total, 2), "stock_value": round(stock, 2) if stock is not None else None,
                  "cash": round(cash, 2) if cash is not None else None,
                  "cash_ratio": round(cash / total, 4) if total and cash is not None else None},
        "display": f"Total account assets {total:,.0f}, including securities {stock:,.0f} and cash {cash:,.0f}"
                   if None not in (stock, cash) else f"Total account assets {total:,.0f}",
        "evidence": [f"bm453_daily.asset_total={total}", f"bm453_daily.stock_total={stock}",
                     f"bm453_daily.money_remain={cash}", f"bm453_daily.snapshot_date={last.get('snapshot_date')}"],
        "confidence": 0.95,
    }


def asset_class_allocation(acct):
    as_of, book = _latest_book(acct)
    if not book:
        return None
    buckets = defaultdict(float)
    n_pos = 0
    for code, p in book.items():
        mv = _f(p.get("stock_market_value"), 0.0) or 0.0
        buckets[classify_instrument(code, p.get("stock_name"))] += mv
        n_pos += 1
    stock_total = sum(buckets.values())
    if stock_total <= 0:
        return None
    shares = {k: round(v / stock_total, 4) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}
    zh = {"single_name_equity": "Individual stocks", "thematic_equity": "Theme/sector ETFs", "index_equity": "Broad-based ETF",
          "leveraged": "Leveraged/inverse products", "fixed_income": "Fixed income/money market", "commodity": "Commodity"}
    top = "、".join(f"{zh.get(k, k)} {v * 100:.0f}%" for k, v in list(shares.items())[:4])
    return {
        "value": {"as_of": as_of, "by_asset_class_within_equity": shares, "positions": n_pos},
        "display": f"Position composition as of {as_of}: {top} (share of securities market value)",
        "evidence": [f"bm453_daily[{as_of}].holdings market value of {n_pos} instruments"],
        "confidence": 0.9,
    }


def strategy_bucket_structure(acct):
    """How the book is organised: core index vs thematic vs satellite bets."""
    as_of, book = _latest_book(acct)
    if not book:
        return None
    rows = [(code, classify_instrument(code, p.get("stock_name")),
             _f(p.get("stock_market_value"), 0.0) or 0.0, _f(p.get("profit_loss_total")))
            for code, p in book.items()]
    total = sum(r[2] for r in rows) or 1.0
    core = sum(r[2] for r in rows if r[1] in ("index_equity", "fixed_income")) / total
    directional = sum(r[2] for r in rows if r[1] in ("single_name_equity", "thematic_equity")) / total
    leveraged = sum(r[2] for r in rows if r[1] == "leveraged") / total
    label = ("Use leverage/derivatives to assume the primary risk exposure" if leveraged >= 0.2 else
             "Primarily focused on directional individual stocks/themes, lacking broad-based or fixed-income core holdings" if core < 0.15 and directional >= 0.6 else
             "Use broad-based/fixed income as core holdings, with satellite positions for directional allocation" if core >= 0.3 else
             "Core and satellite positions are relatively balanced")
    return {
        "value": {"as_of": as_of, "core_share": round(core, 4),
                  "directional_share": round(directional, 4),
                  "leveraged_share": round(leveraged, 4), "positions": len(rows)},
        "display": label + f"(As of {as_of})",
        "evidence": [f"bm453_daily[{as_of}].holdings grouped and aggregated by instrument ({len(rows)} instruments)"],
        "confidence": 0.75,
    }


def largest_position_concentration(acct):
    if not acct:
        return None
    series = []
    for d in acct:
        book = d.get("holdings") or {}
        total = sum(_f(p.get("stock_market_value"), 0.0) or 0.0 for p in book.values())
        if total and book:
            top_code, top_mv = max(
                ((c, _f(p.get("stock_market_value"), 0.0) or 0.0) for c, p in book.items()),
                key=lambda kv: kv[1])
            series.append((d.get("snapshot_date"), top_code, top_mv / total))
    if not series:
        return None
    last_date, last_code, last_share = series[-1]
    avg = sum(s[2] for s in series) / len(series)
    max_seen = max(s[2] for s in series)
    level = ("Extremely high" if avg >= 0.5 else "Higher" if avg >= 0.3 else "Moderate" if avg >= 0.15 else "Diversification")
    return {
        "value": {"latest_top_holding": last_code, "latest_share": round(last_share, 4),
                  "avg_top1_share": round(avg, 4), "max_top1_share": round(max_seen, 4)},
        "display": f"Average maximum single position accounted for {avg * 100:.0f}% of securities market value (concentration {level}); most recently, {last_code} accounted for {last_share * 100:.0f}%",
        "evidence": [f"bm453_daily.holdings daily top1 market value share ({len(series)} days)",
                     f"position.max_num_positions"],
        "confidence": 0.9,
    }


def sector_concentration(acct):
    as_of, last = _latest_book(acct)
    if not last:
        return None
    sec = defaultdict(float)
    unknown = 0.0
    for code, p in last.items():
        mv = _f(p.get("stock_market_value"), 0.0) or 0.0
        info = ticker_info(code)
        if info["sector"] == "Other":
            unknown += mv
            sec["未分类"] += mv
        else:
            sec[info["sector"]] += mv
    total = sum(sec.values()) or 1.0
    shares = {k: round(v / total, 4) for k, v in sorted(sec.items(), key=lambda kv: -kv[1])}
    top_sector, top_share = next(iter(shares.items()))
    coverage = 1 - unknown / total
    return {
        "value": {"as_of": as_of, "by_sector": shares, "top_sector": top_sector,
                  "top_sector_share": top_share, "sector_map_coverage": round(coverage, 4)},
        "display": f"Sector concentration: {top_sector} accounts for {top_share * 100:.0f}% (as of {as_of})"
                   + ("; The static industry table does not cover some instruments, so this value may underestimate concentration" if coverage < 0.8 else ""),
        "evidence": [f"bm453_daily[{as_of}].holdings mapped by static industry table, coverage {coverage:.0%}"],
        "confidence": round(0.85 * coverage, 3),
    }


def _fmt_seq(seq):
    """Presentable text for a container, never a Python repr.

    An f-string interpolation of a list/dict prints Python syntax into the UI
    (`['SOXL']`, `{'SOXL': 8}`). That syntax is Python-specific, so it reads as a
    defect to the user and to any non-Python consumer of the profile.
    """
    items = [str(x) for x in (seq or []) if x]
    return "、".join(items) if items else "None"


def _fmt_counts(mapping, sep="、"):
    """`{"SOXL": 8}` -> "SOXL×8". Never a Python repr."""
    return sep.join(f"{k}×{v}" for k, v in (mapping or {}).items()) or "None"


def leverage_and_derivatives_exposure(acct, trades, intent):
    """Leveraged/inverse products held or traded, plus option activity."""
    _as_of, book = _latest_book(acct)
    held_lev = [code for code, p in book.items()
                if _is_leveraged(code, p.get("stock_name"))]
    traded_lev = Counter()
    for t in trades or []:
        if _is_leveraged(t.get("ticker"), None):
            traded_lev[(t.get("ticker") or "").upper()] += 1
    opt_views = 0
    opt_trades = 0
    if intent and intent.get("available"):
        for row in intent.get("event_mix") or []:
            if row.get("event_type") == "view_options":
                opt_views = row.get("count") or 0
    for t in trades or []:
        if re.fullmatch(r"[A-Z]{1,5}\d{6}[CP]\d+", (t.get("ticker") or "").upper()):
            opt_trades += 1
    if not held_lev and not traded_lev and not opt_views and not opt_trades:
        return {
            "value": {"leveraged_held": [], "leveraged_traded": {}, "option_views": 0, "option_trades": 0},
            "display": "No use of leveraged/inverse products or options observed",
            "evidence": ["Neither bm453_daily.holdings nor bm452_trades contains leveraged/option instruments",
                         "bm433 no option-chain viewing behavior"],
            "confidence": 0.8,
        }
    parts = []
    if held_lev:
        parts.append("Hold " + "、".join(held_lev[:4]))
    if traded_lev:
        parts.append("Trade " + "、".join(traded_lev.keys()))
    if opt_views:
        parts.append(f"Viewed option chains {opt_views} times")
    if opt_trades:
        parts.append(f"Option trades {opt_trades}")
    return {
        "value": {"leveraged_held": held_lev, "leveraged_traded": dict(traded_lev),
                  "option_views": opt_views, "option_trades": opt_trades},
        "display": "Leverage and derivatives use:" + "；".join(parts),
        "evidence": [f"bm453_daily.holdings leveraged/inverse products: {_fmt_seq(held_lev)}",
                     f"bm452_trades leveraged product trades: {_fmt_counts(traded_lev)}",
                     f"bm433 view_options=×{opt_views}"],
        "confidence": 0.85,
    }


def contribution_and_withdrawal_pattern(acct, trades):
    """External cash flow pattern (detection lives in app/account.py)."""
    if not acct or len(acct) < 3:
        return None
    flows = external_flows(acct, trades)
    if not flows:
        return {
            "value": {"events": [], "deposits": 0, "withdrawals": 0,
                      "net": 0.0, "observation_days": len(acct)},
            "display": "No verifiable net deposits or withdrawals identified during the observation window (inferred basis: no trading days and unchanged holdings)",
            "evidence": [f"bm453_daily {len(acct)}-day asset series, with daily Δtotal assets−Δsecurities market value≈0"],
            "confidence": 0.6,
        }
    dep = [f for f in flows if f["direction"] == "in"]
    wd = [f for f in flows if f["direction"] == "out"]
    net = sum(f["amount"] for f in flows)
    first = _f(acct[0].get("asset_total")) or 0
    material = bool(first) and abs(net) > 0.05 * abs(first)
    note = ("Account deposits and withdrawals have a significant impact on total assets; measuring investment performance by changes in total assets would be severely distorted"
            if material else "Account returns are affected by capital inflows and outflows and should not be measured directly by changes in total assets")
    return {
        "value": {"events": flows[:12], "deposits": len(dep), "withdrawals": len(wd),
                  "net": round(net, 2), "observation_days": len(acct),
                  "material": material},
        "display": (f"Identified {len(dep)} deposits and {len(wd)} withdrawals (net {'+' if net >= 0 else ''}{net:,.0f}); {note}."),
        "evidence": [f"On non-trading days in bm453_daily, Δasset_total−Δstock_total was non-zero ({len(flows)} days)",
                     f"example {flows[0]['date']} "
                     f"{'deposit' if flows[0]['amount'] > 0 else 'withdrawal'} "
                     f"{abs(flows[0]['amount']):,.0f}"],
        "confidence": 0.6,
    }


def drawdown_experience(acct, trades=None):
    """Worst peak-to-trough on the flow-adjusted curve.

    Uses the same time-weighted basis as evidence.asset so the KYC leaf and the
    risk evidence can never disagree about the same drawdown.
    """
    if not acct or len(acct) < 2:
        return None
    adj = adjusted_equity_series(acct, trades)
    dates, values = adj["dates"], adj["values"]
    if len(values) < 2 or not adj.get("reliable", True):
        # covering too few days to claim a drawdown: report a gap, not 0%
        return None
    worst, worst_date, peak_at, recover = max_drawdown(dates, values)
    if not worst_date:
        return None
    relief = worst <= -0.2
    return {
        "value": {"max_drawdown": worst, "worst_date": worst_date,
                  "peak_value": peak_at, "recovered_by": recover,
                  "basis": "flow_adjusted" if adj["flow_count"] else "raw",
                  "external_flow_events": adj["flow_count"]},
        "display": f"Maximum account drawdown during the observation window: {worst * 100:.1f}% (reached on {worst_date})"
                   + (", subsequently recovered" if recover else ", has not yet returned to the previous high")
                   + ("; The impact of external deposits and withdrawals has been excluded" if adj["flow_count"] else "")
                   + ("; Continued holding despite a relatively large drawdown, indicating a level of pressure actually endured" if relief else ""),
        "evidence": [f"bm453_daily time-weighted net asset value series: {len(values)} days ({dates[0]} to {dates[-1]})",
                     f"Removed {adj['flow_count']} external deposits/withdrawals (net {adj['flow_net']:,.0f})"],
        "confidence": 0.8 if adj["flow_count"] else 0.85,
    }


def investable_assets_band(acct):
    if not acct:
        return None
    total = _f(acct[-1].get("asset_total"))
    if total is None:
        return None
    band = _band(total, [1_000, 10_000, 50_000, 250_000, 1_000_000],
                 ["<1 thousand", "1,000-10,000", "10,000-50,000", "50,000-250,000", "250,000-1,000,000", ">1 million"])
    return {
        "value": {"latest_asset_total": round(total, 2), "band": band, "currency": "USD"},
        "display": f"Investable assets fall within the {band} range (platform account basis)",
        "evidence": [f"bm453_daily.asset_total={total}", "Only covers accounts visible on this platform; external assets are not visible"],
        "confidence": 0.9,
    }


# ---------- registry ----------

DETERMINISTIC = {
    "account_balances_and_cash": lambda acct, tr, it: account_balances_and_cash(acct),
    "asset_class_allocation": lambda acct, tr, it: asset_class_allocation(acct),
    "strategy_bucket_structure": lambda acct, tr, it: strategy_bucket_structure(acct),
    "largest_position_concentration": lambda acct, tr, it: largest_position_concentration(acct),
    "sector_concentration": lambda acct, tr, it: sector_concentration(acct),
    "leverage_and_derivatives_exposure": lambda acct, tr, it: leverage_and_derivatives_exposure(acct, tr, it),
    "contribution_and_withdrawal_pattern": lambda acct, tr, it: contribution_and_withdrawal_pattern(acct, tr),
    "drawdown_experience": lambda acct, tr, it: drawdown_experience(acct, tr),
    "investable_assets_band": lambda acct, tr, it: investable_assets_band(acct),
}


def build_deterministic_features(snap, intent=None):
    """{feature_key: {value, display, evidence, confidence, coverage, method}}"""
    acct = snap.get("account_daily") or []
    trades = snap.get("trades_6m") or []
    out = {}
    for key, fn in DETERMINISTIC.items():
        try:
            r = fn(acct, trades, intent)
        except Exception as e:                                        # noqa: BLE001
            r = None
        if r is None:
            r = {"value": None, "display": "Insufficient existing data to calculate this feature",
                 "evidence": [], "confidence": 0.0}
        out[key] = {**r, "coverage": "deterministic", "method": "rule"}
    from .english_output import englishize
    return englishize(out)
