"""Shared taxonomy: entity linking, sector/theme mapping, keyword matching.

Used by both the evidence builder and the BM433 behavior engine so a ticker is
classified identically everywhere. Static tables on purpose: deterministic and
auditable, no network, no LLM.
"""
import re
from datetime import datetime, timezone

# Static sector/theme map for the tickers observed in this dataset.
# Unknown tickers degrade to "Other" rather than being guessed.
SECTOR_MAP = {
    "NVDA": ("Semiconductors", "AI"), "AMD": ("Semiconductors", "AI"),
    "INTC": ("Semiconductors", "AI"), "MU": ("Semiconductors", "Memory/HBM"),
    "SNDK": ("Semiconductors", "Memory/HBM"), "SOC": ("Semiconductors", "Memory/HBM"),
    "TSM": ("Semiconductors", "AI"), "AVGO": ("Semiconductors", "AI"),
    "MRVL": ("Semiconductors", "AI"), "QCOM": ("Semiconductors", "AI"),
    "TSLA": ("EV/Auto", "Consumer"), "SPCX": ("Space", "Growth"),
    "RKLB": ("Space", "Growth"), "NBIS": ("AI Infra", "AI"),
    "LITE": ("Optical/Networking", "AI"), "COHR": ("Optical/Networking", "AI"),
    "SMCI": ("AI Infra", "AI"), "ARM": ("Semiconductors", "AI"),
    "GOOGL": ("Mega-cap Tech", "AI"), "META": ("Mega-cap Tech", "AI"),
    "AAPL": ("Mega-cap Tech", "Consumer"), "MSFT": ("Mega-cap Tech", "AI"),
    "AMZN": ("Mega-cap Tech", "Consumer"), "MSTR": ("Crypto Proxy", "Crypto"),
    "COIN": ("Crypto Proxy", "Crypto"), "PLTR": ("Software/AI", "AI"),
    "QQQ": ("Index ETF", "Broad"), "SPY": ("Index ETF", "Broad"),
    "IWM": ("Index ETF", "Broad"), "DRAM": ("Theme ETF", "Memory/HBM"),
    "SOXL": ("Leveraged ETF", "Semiconductors"), "SOXS": ("Leveraged ETF", "Semiconductors"),
    "TQQQ": ("Leveraged ETF", "Broad"), "SQQQ": ("Leveraged ETF", "Broad"),
    "NVDL": ("Leveraged ETF", "AI"), "USDY": ("Yield/Income", "Income"),
    "SKHY": ("Semiconductors", "Memory/HBM"), "SSNLF": ("Semiconductors", "Memory/HBM"),
    "OKLO": ("Nuclear/Energy", "Energy"), "IREN": ("Crypto Proxy", "Crypto"),
    "HOOD": ("Brokerage/Fintech", "Growth"), "IONQ": ("Quantum", "Growth"),
    "BTC": ("Crypto", "Crypto"), "ETH": ("Crypto", "Crypto"), "USDT": ("Crypto", "Crypto"),
    # tickers observed in the BM433 event vocabulary
    "SGOV": ("Treasury ETF", "Income"), "DIA": ("Index ETF", "Broad"),
    "VOO": ("Index ETF", "Broad"), "VTI": ("Index ETF", "Broad"),
    "RKLX": ("Space", "Growth"), "QBTS": ("Quantum", "Growth"),
    "RGTI": ("Quantum", "Growth"), "OPEN": ("Real Estate/PropTech", "Growth"),
    "MP": ("Materials", "Energy"), "UUUU": ("Materials", "Energy"),
    # crypto assets seen as uppercase tokens in news headlines
    "XRP": ("Crypto", "Crypto"), "SOL": ("Crypto", "Crypto"),
    "ADA": ("Crypto", "Crypto"), "DOGE": ("Crypto", "Crypto"),
    "AVAX": ("Crypto", "Crypto"), "LINK": ("Crypto", "Crypto"),
    "BNB": ("Crypto", "Crypto"), "TON": ("Crypto", "Crypto"),
    # names appearing as bare tokens in the news corpus
    "NVMI": ("Semiconductors", "AI"), "PLTR": ("Software/AI", "AI"),
    "FUTU": ("Brokerage/Fintech", "Growth"), "BABA": ("Mega-cap Tech", "Consumer"),
    "NIO": ("EV/Auto", "Consumer"), "RIVN": ("EV/Auto", "Consumer"),
    "LCID": ("EV/Auto", "Consumer"), "RDDT": ("Social/Media", "Growth"),
    "SNAP": ("Social/Media", "Growth"), "UBER": ("Platform", "Consumer"),
    "ABNB": ("Platform", "Consumer"), "SHOP": ("Platform", "Growth"),
    "CRWV": ("AI Infra", "AI"), "CRWD": ("Cybersecurity", "AI"),
    "NET": ("Cybersecurity", "AI"), "DDOG": ("Software/AI", "AI"),
    "SNOW": ("Software/AI", "AI"), "NOW": ("Software/AI", "AI"),
}

# Company / product names as they actually appear in BM433 news and BM428
# conversation titles (mostly Chinese). Mapped to a ticker so the sector/theme
# lookup stays shared.
NAME_ALIASES = {
    "英伟达": "NVDA", "nvidia": "NVDA", "海力士": "SKHY", "sk hynix": "SKHY",
    "美光": "MU", "micron": "MU", "三星": "SSNLF", "samsung": "SSNLF",
    "英特尔": "INTC", "intel": "INTC", "苹果": "AAPL", "apple": "AAPL",
    "特斯拉": "TSLA", "tesla": "TSLA", "亚马逊": "AMZN", "amazon": "AMZN",
    "谷歌": "GOOGL", "alphabet": "GOOGL", "微软": "MSFT", "microsoft": "MSFT",
    "台积电": "TSM", "比特币": "BTC", "bitcoin": "BTC", "以太坊": "ETH",
    "tether": "USDT", "lumentum": "LITE", "coherent": "COHR", "cohr": "COHR",
    "oklo": "OKLO", "iren": "IREN", "hood": "HOOD", "ionq": "IONQ", "shein": "SHEIN",
    "甲骨文": "ORCL", "oracle": "ORCL", "博通": "AVGO", "broadcom": "AVGO",
    "超微": "AMD", "高通": "QCOM", "qualcomm": "QCOM", "奈飞": "NFLX",
    "netflix": "NFLX", "礼来": "LLY", "波音": "BA", "boeing": "BA",
    # English company names that dominate the BM433 news corpus
    "meta platforms": "META", "palantir": "PLTR", "broadcom": "AVGO",
    "micron technology": "MU", "advanced micro": "AMD", "salesforce": "CRM",
    "goldman sachs": "GS", "morgan stanley": "MS", "jp morgan": "JPM",
    "jpmorgan": "JPM", "bank of america": "BAC", "walmart": "WMT",
    "disney": "DIS", "pfizer": "PFE", "exxon": "XOM", "chevron": "CVX",
    "alibaba": "BABA", "baidu": "BIDU", "nio": "NIO", "rivian": "RIVN",
    "uber": "UBER", "airbnb": "ABNB", "shopify": "SHOP", "reddit": "RDDT",
    "crowdstrike": "CRWD", "snowflake": "SNOW", "datadog": "DDOG",
    "cloudflare": "NET", "servicenow": "NOW", "coreweave": "CRWV",
}

# Macro / geopolitical entities. The BM433 news stream is English while the
# BM428 conversation titles are Chinese, so both spellings are needed — using
# only Chinese here silently dropped the largest news topics in the corpus
# (Trump, Iran, Fed, CPI, China...).
MACRO_ENTITIES = {
    "Tariff/Trade": ["关税", "tariff", "trade war", "贸易战", "sanction", "制裁"],
    "Election/Politics": ["大选", "election", "trump", "biden", "白宫", "white house",
                         "congress", "国会", "kimmel"],
    "Inflation/CPI": ["通胀", "cpi", "pce", "inflation", "物价"],
    "Employment": ["非农", "就业", "payroll", "unemployment", "失业"],
    "Fed/Rates": ["美联储", "fed", "fomc", "利率决议", "powell", "鲍威尔", "warsh",
                  "rate cut", "rate hike", "降息", "加息"],
    "Oil/Energy": ["油价", "原油", "oil", "opec", "crude", "天然气", "energy crisis"],
    "Geopolitics": ["伊朗", "iran", "伊朗", "俄罗斯", "russia", "乌克兰", "ukraine",
                    "israel", "以色列", "war", "冲突", "地缘", "hormuz", "霍尔木兹",
                    "nato", "军事"],
    "China": ["中国", "china", "北京", "beijing", "人民币", "中概"],
    "Japan": ["日本", "japan", "日元", "boj", "日银"],
    "Earnings Season": ["财报季", "earnings season", "earnings beat", "业绩超预期"],
}

# Non-ticker topical themes worth tracking in a financial profile.
TOPIC_KEYWORDS = {
    "FX": ["美元", "人民币", "汇率", "dollar", "yuan", "cny", "fx", "usdcny", "forex"],
    "Rates": ["收益率", "美债", "国债", "yield", "treasury", "fed", "美联储", "利率",
              "加息", "降息", "fomc", "warsh", "powell", "鲍威尔", "bond"],
    "Crypto": ["比特币", "加密货币", "crypto", "bitcoin", "tether", "稳定币", "以太坊",
               "xrp", "ripple", "solana", "ethereum", "memecoin"],
    "Index": ["纳斯达克", "标普", "nasdaq", "s&p", "指数", "道琼斯", "dow"],
    "Earnings": ["财报", "业绩", "earnings", "guidance", "指引", "季报", "盈利",
                 "revenue", "profit", "estimates"],
    "Energy": ["油价", "原油", "oil", "opec", "天然气", "crude"],
    "Geopolitics": ["伊朗", "俄罗斯", "俄乌", "地缘", "war", "conflict", "军事", "关税",
                    "tariff", "iran", "israel", "russia", "ukraine", "hormuz"],
    "Memory/HBM": ["内存", "存储", "hbm", "dram", "memory"],
    "AI": ["人工智能", "ai算力", "mag 7", "ai", "算力", "datacenter", "gpu"],
    "Space": ["spacex", "star ship", "starship", "火箭", "卫星", "rocket", "space"],
    "Quantum": ["quantum", "量子"],
    "IPO": ["ipo", "上市", "新股"],
}

# Macro / event entities that matter to investment logic.

ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{1,6}\b")
_COMPACT_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(.*)$")

# Issuer domicile for non-US names that actually appear in the holdings data.
# Used only for a coarse geographic-exposure proxy, never as a precise
# attribution: these are US-listed ADRs / foreign ordinaries, so the *listing*
# is American while the *exposure* is not. Anything absent is treated as US
# (the platform's default universe), which the output states explicitly.
COUNTRY = {
    "TSM": "Taiwan", "BABA": "China", "PDD": "China", "JD": "China", "NIO": "China",
    "BIDU": "China", "TCEHY": "China", "SHEIN": "China", "FUTU": "China",
    "SKHY": "Korea", "SSNLF": "Korea", "NOK": "Finland", "ERIC": "Sweden",
    "ASML": "Netherlands", "ARM": "UK", "BTI": "UK", "BP": "UK", "SHEL": "UK",
    "AZN": "UK", "GSK": "UK", "UL": "UK", "RIO": "UK", "SPOT": "Sweden",
    "SONY": "Japan", "TM": "Japan", "HMC": "Japan", "SAP": "Germany",
    "SNY": "France", "DEO": "UK", "NVS": "Switzerland", "MELI": "Argentina",
    "NU": "Brazil", "VALE": "Brazil", "GRAB": "Singapore", "SE": "Singapore",
    "INFY": "India", "WIT": "India", "CPNG": "Korea", "STLA": "Netherlands",
}
US_DEFAULT = "United States"


def country_of(ticker):
    """Issuer domicile, defaulting to the platform's US universe."""
    return COUNTRY.get((ticker or "").upper(), US_DEFAULT)


def parse_ts(s):
    """Parse the several timestamp shapes that appear in the BM streams.

    Handles ISO with offset (`2026-08-11T20:31:00-05:00`), ISO without
    (`2026-08-13T12:00`), date-only (`2026-08-13`) and the compact collector
    form (`20260813`, `20260813T12:00`).

    Timezone-aware values are converted to UTC and returned naive, so every
    timestamp in the pipeline is directly comparable: mixing naive and aware
    datetimes is a hard TypeError in Python and was silently dropping rows.
    """
    if not s:
        return None
    s = str(s).strip()
    m = _COMPACT_TS_RE.match(s)
    if m:
        s = f"{m.group(1)}-{m.group(2)}-{m.group(3)}{m.group(4)}"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def ticker_info(t):
    """(sector, theme) for a ticker; unknown tickers degrade to Other."""
    sec, theme = SECTOR_MAP.get((t or "").upper(), ("Other", "Other"))
    return {"sector": sec, "theme": theme}


def kw_hit(low, kw):
    """CJK keywords match as substrings; latin keywords need word boundaries
    (otherwise 'ai' hits 'aime' and 'war' hits 'Warsh')."""
    if ZH_RE.search(kw):
        return kw in low
    return re.search(rf"\b{re.escape(kw)}\b", low) is not None


def extract_entities(text, ticker_whitelist=None):
    """Link free text (news title / conversation title) to tradable entities.

    Returns {"tickers": [...], "themes": [...], "sectors": [...], "macro": [...]}.
    Tickers come from an explicit whitelist when we have one (the user's own
    viewed universe), otherwise from the known SECTOR_MAP keys + name aliases.
    """
    t = (text or "")
    low = t.lower()
    found = []

    # 1. alias names first (英伟达 -> NVDA) so the theme follows the company
    for alias, tk in NAME_ALIASES.items():
        if alias in low:
            found.append(tk)
    # 2. bare uppercase ticker tokens
    for tok in _TICKER_TOKEN_RE.findall(t):
        tk = tok.upper()
        if tk in SECTOR_MAP:
            found.append(tk)
        elif ticker_whitelist and tk in ticker_whitelist:
            found.append(tk)

    tickers = list(dict.fromkeys(found))
    themes, sectors = [], []
    for tk in tickers:
        info = ticker_info(tk)
        if info["theme"] != "Other":
            themes.append(info["theme"])
        if info["sector"] != "Other":
            sectors.append(info["sector"])
    for theme, kws in TOPIC_KEYWORDS.items():
        if any(kw_hit(low, k) for k in kws):
            themes.append(theme)
    macro = [m for m, kws in MACRO_ENTITIES.items() if any(kw_hit(low, k) for k in kws)]
    return {"tickers": tickers, "themes": list(dict.fromkeys(themes)),
            "sectors": list(dict.fromkeys(sectors)), "macro": macro}


def strip_crypto_suffix(code):
    """BTCUSDT -> BTC, but leave real tickers alone."""
    c = (code or "").upper()
    for suf in ("USDT", "USDC", "BUSD", "USD"):
        if c.endswith(suf) and len(c) > len(suf) + 1:
            base = c[: -len(suf)]
            if base in SECTOR_MAP:
                return base
            return base
    return c
