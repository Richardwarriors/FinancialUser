"""Point-in-time Yahoo market context for rolling prediction samples.

Raw Yahoo bars are cached separately from the compact object consumed by a
predictor.  Every feature is computed from at most the seven market sessions at
or before the sample cutoff; a target-day bar is never read.
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BENCHMARKS = ("SPY", "QQQ")
WINDOW_TRADING_DAYS = 7
MOVERS_K = 3
WATCHLIST_K = 6
BLOCK_BUDGET = {
    "regime": 1200,
    "portfolio": 900,
    "movers": 900,
    "watchlist": 1200,
}

_OPTION = re.compile(r"^([A-Z]{1,6})(\d{6}|\d{8})([CP])(\d+(?:\.\d+)?)$")
_CRYPTO = re.compile(r"^([A-Z]{2,8})(USD|USDT|USDC)$")
_USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; rolling-market/1.0)"}


def underlying_symbol(symbol: str) -> tuple[str, str | None]:
    """Return a priceable symbol and whether it is an underlying proxy."""
    raw = (symbol or "").strip().upper()
    match = _OPTION.match(raw)
    return (match.group(1), "underlying") if match else (raw, None)


def yahoo_symbol(symbol: str) -> str:
    """Map feed spellings to Yahoo chart endpoint spellings."""
    match = _CRYPTO.match(symbol)
    if match:
        return f"{match.group(1)}-USD"
    if re.fullmatch(r"[A-Z]{1,6}\.[A-Z]{1,2}", symbol):
        return symbol.replace(".", "-")
    return symbol


def _safe_name(symbol: str) -> str:
    return urllib.parse.quote(symbol, safe="") + ".json"


class YahooMarketCache:
    """Disk cache around Yahoo's unauthenticated daily chart endpoint."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def path_for(self, symbol: str) -> Path:
        return self.cache_dir / _safe_name(symbol)

    def load(self, symbol: str) -> list[dict]:
        path = self.path_for(symbol)
        if not path.is_file():
            return []
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        rows = [row for row in (doc.get("series") or [])
                if row.get("date") and row.get("close") is not None]
        return sorted(rows, key=lambda row: row["date"])

    def fetch(self, symbol: str, start: date, end: date, timeout: int = 30,
              retries: int = 3) -> list[dict]:
        """Fetch and cache daily OHLCV. Network access is explicit via this method."""
        mapped = yahoo_symbol(symbol)
        period1 = int(datetime.combine(start, datetime.min.time(),
                                       tzinfo=timezone.utc).timestamp())
        # Yahoo's period2 is exclusive.
        period2 = int(datetime.combine(end + timedelta(days=1), datetime.min.time(),
                                       tzinfo=timezone.utc).timestamp())
        query = urllib.parse.urlencode({"period1": period1, "period2": period2,
                                        "interval": "1d"})
        url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{urllib.parse.quote(mapped)}?{query}")
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, headers=_USER_AGENT)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                result = ((body.get("chart") or {}).get("result") or [None])[0]
                if not result:
                    detail = ((body.get("chart") or {}).get("error") or {}).get(
                        "description", "empty Yahoo result")
                    raise ValueError(detail)
                rows = _series_from_yahoo(result)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                doc = {"symbol": symbol, "yahoo_symbol": mapped, "series": rows,
                       "fetched_at": datetime.now(timezone.utc).isoformat()}
                self.path_for(symbol).write_text(json.dumps(doc), encoding="utf-8")
                return rows
            except (OSError, ValueError, json.JSONDecodeError,
                    urllib.error.URLError) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    # Keep retry deterministic and short; callers control outer pacing.
                    continue
        raise RuntimeError(f"Yahoo fetch failed for {symbol}: {last_error}")


def _series_from_yahoo(result: dict) -> list[dict]:
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjusted = (((result.get("indicators") or {}).get("adjclose") or [{}])[0]
                .get("adjclose") or [])
    rows = []
    for index, timestamp in enumerate(timestamps):
        closes = quote.get("close") or []
        close = closes[index] if index < len(closes) else None
        if close is None:
            continue
        def value(key):
            values = quote.get(key) or []
            return values[index] if index < len(values) else None
        rows.append({
            "date": datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat(),
            "close": close,
            "adjclose": adjusted[index] if index < len(adjusted) else close,
            "high": value("high"),
            "low": value("low"),
            "volume": value("volume"),
        })
    return rows


def bars_at_cutoff(series: list[dict], cutoff: date,
                   window: int = WINDOW_TRADING_DAYS) -> list[dict]:
    """Return exactly the newest available trading bars, never after cutoff."""
    eligible = [row for row in series
                if row.get("date") and date.fromisoformat(row["date"]) <= cutoff]
    eligible.sort(key=lambda row: row["date"])
    return eligible[-window:]


def market_features(series: list[dict], cutoff: date) -> dict:
    """Compact features from no more than seven point-in-time daily bars."""
    bars = bars_at_cutoff(series, cutoff)
    if not bars:
        return {"status": "unavailable", "reason": "no bar at or before cutoff",
                "bars_used": 0, "window_trading_days": WINDOW_TRADING_DAYS}
    closes = [float(row["close"]) for row in bars]
    returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))
               if closes[i - 1]]
    last = bars[-1]
    volumes = [float(row["volume"]) for row in bars if row.get("volume") is not None]
    volume_z = None
    if len(volumes) >= 3:
        spread = st.pstdev(volumes)
        volume_z = (volumes[-1] - st.fmean(volumes)) / spread if spread else 0.0
    range_pct = None
    if last.get("high") is not None and last.get("low") is not None and closes[-1]:
        range_pct = (last["high"] - last["low"]) / closes[-1]
    return {
        "status": "available" if len(bars) == WINDOW_TRADING_DAYS else "limited_history",
        "as_of_date": last["date"],
        "window_start_date": bars[0]["date"],
        "window_trading_days": WINDOW_TRADING_DAYS,
        "bars_used": len(bars),
        "close": closes[-1],
        "return_1d": _round(returns[-1]) if returns else None,
        "return_window": _round(closes[-1] / closes[0] - 1) if len(closes) > 1 else None,
        "realized_vol_annualized": (_round(st.stdev([math.log(closes[i] / closes[i - 1])
                                                     for i in range(1, len(closes))])
                                             * math.sqrt(252))
                                     if len(closes) > 2 else None),
        "range_pct": _round(range_pct),
        "volume_z_window": _round(volume_z),
    }


def build_market_context(cutoff: date, positions: list[dict], watchlist: list[dict],
                         loader, benchmarks: tuple[str, ...] = BENCHMARKS) -> dict:
    """Build prompt-ready market context without exposing raw per-ticker bars.

    `loader(symbol)` must return cached OHLCV rows.  Positions accept `ticker` and
    `market_value`; watchlist rows accept `ticker` and optionally `views`.
    """
    requested: dict[str, dict] = {}
    for ticker in list(benchmarks) + [p.get("ticker") for p in positions] + [
            row.get("ticker") for row in watchlist]:
        if not ticker:
            continue
        priced, proxy = underlying_symbol(ticker)
        meta = requested.setdefault(priced, {"source_tickers": [], "proxy_type": proxy})
        if ticker not in meta["source_tickers"]:
            meta["source_tickers"].append(ticker)
        if proxy:
            meta["proxy_type"] = proxy

    features = {}
    missing = []
    for symbol in sorted(requested):
        rows = loader(symbol) or []
        feature = market_features(rows, cutoff)
        features[symbol] = feature
        if feature["status"] == "unavailable":
            missing.append(symbol)

    regime = {symbol: features.get(symbol, {"status": "unavailable"})
              for symbol in benchmarks}
    portfolio = _portfolio_block(positions, features)
    movers = _movers_block(positions, features)
    watch = _watchlist_block(watchlist, positions, features)
    blocks = {"regime": regime, "portfolio": portfolio,
              "movers": movers, "watchlist": watch}
    budget = {name: {"chars": _json_len(block), "limit": BLOCK_BUDGET[name],
                     "over_budget": _json_len(block) > BLOCK_BUDGET[name]}
              for name, block in blocks.items()}
    return {
        "as_of_date": cutoff.isoformat(),
        **blocks,
        "coverage": {
            "symbols_requested": len(requested),
            "symbols_available": len(requested) - len(missing),
            "symbols_missing": missing,
            "proxy_mappings": [
                {"source": source, "priced_as": symbol, "proxy_type": "underlying"}
                for symbol, meta in sorted(requested.items())
                for source in meta["source_tickers"] if source != symbol
            ],
        },
        "budget": budget,
        "point_in_time": {
            "clean": all((f.get("as_of_date") or "") <= cutoff.isoformat()
                         for f in features.values()),
            "rule": "all market bars are on or before cutoff; newest seven bars only",
        },
    }


def _feature_for_ticker(ticker: str, features: dict) -> dict:
    return features.get(underlying_symbol(ticker)[0]) or {"status": "unavailable"}


def _portfolio_block(positions: list[dict], features: dict) -> dict:
    usable, missing = [], []
    for position in positions:
        ticker, value = position.get("ticker"), position.get("market_value")
        feature = _feature_for_ticker(ticker or "", features)
        if not ticker or not isinstance(value, (int, float)) or value <= 0:
            continue
        if feature.get("status") not in ("available", "limited_history"):
            missing.append(ticker)
            continue
        usable.append((feature, float(value)))
    if not usable:
        return {"status": "unavailable", "priced_positions": 0,
                "missing_prices": sorted(missing)}
    total = sum(weight for _, weight in usable)
    def weighted(field):
        pairs = [(feature.get(field), weight) for feature, weight in usable
                 if feature.get(field) is not None]
        denominator = sum(weight for _, weight in pairs)
        return _round(sum(value * weight for value, weight in pairs) / denominator) \
            if denominator else None
    up = sum(1 for feature, _ in usable if (feature.get("return_window") or 0) > 0)
    returns = [feature["return_window"] for feature, _ in usable
               if feature.get("return_window") is not None]
    return {"status": "available", "priced_positions": len(usable),
            "unpriced_positions": len(missing), "missing_prices": sorted(missing),
            "priced_market_value": _round(total),
            "weighted_return_1d": weighted("return_1d"),
            "weighted_return_window": weighted("return_window"),
            "weighted_realized_vol": weighted("realized_vol_annualized"),
            "share_up_window": _round(up / len(usable)),
            "return_dispersion": _round(st.pstdev(returns)) if len(returns) > 1 else None}


def _movers_block(positions: list[dict], features: dict) -> dict:
    rows = []
    for position in positions:
        ticker = position.get("ticker")
        feature = _feature_for_ticker(ticker or "", features)
        if ticker and feature.get("return_window") is not None:
            priced, proxy = underlying_symbol(ticker)
            rows.append({"ticker": ticker, "priced_as": priced,
                         "proxy_type": proxy, "return_window": feature["return_window"],
                         "realized_vol": feature.get("realized_vol_annualized")})
    if not rows:
        return {"status": "unavailable", "reason": "no priced held position"}
    rows.sort(key=lambda row: row["return_window"])
    return {"status": "available", "worst": rows[:MOVERS_K],
            "best": list(reversed(rows[-MOVERS_K:]))}


def _watchlist_block(watchlist: list[dict], positions: list[dict], features: dict) -> dict:
    held = {position.get("ticker") for position in positions}
    rows, missing = [], []
    for item in watchlist:
        ticker = item.get("ticker")
        if not ticker or ticker in held:
            continue
        feature = _feature_for_ticker(ticker, features)
        priced, proxy = underlying_symbol(ticker)
        row = {"ticker": ticker, "priced_as": priced, "proxy_type": proxy,
               "views": item.get("views", 0), "status": feature.get("status")}
        if feature.get("status") in ("available", "limited_history"):
            row.update({"return_window": feature.get("return_window"),
                        "return_1d": feature.get("return_1d"),
                        "realized_vol": feature.get("realized_vol_annualized")})
            rows.append(row)
        else:
            missing.append(ticker)
    rows.sort(key=lambda row: (-row["views"], row["ticker"]))
    return {"status": "available" if rows or missing else "unavailable",
            "candidates_available": len(rows), "candidates_missing": len(missing),
            "top": rows[:WATCHLIST_K], "missing_prices": sorted(missing)[:WATCHLIST_K]}


def _round(value, digits: int = 6):
    return round(value, digits) if isinstance(value, (int, float)) else value


def _json_len(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
