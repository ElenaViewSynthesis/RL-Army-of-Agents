"""Market-data tools for the finance sub-agents — live Financial Modeling Prep.

Each function calls an FMP ``/stable`` endpoint via :mod:`fmp_client` and
returns a JSON-serializable dict. ADK wraps them as tools automatically; the
first docstring line becomes the tool description the model sees, so keep it
crisp. Endpoint paths match the sibling ``Equity-Research-agent``.

Tools never raise: on a missing key, premium gate, or network failure they
return ``{"error": ...}`` so the model can report the gap rather than crash.
"""

from __future__ import annotations

from .fmp_client import fmp_get, first_record


def get_company_profile(symbol: str) -> dict:
    """Return company profile: name, sector, industry, market cap, description.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return first_record(fmp_get("profile", {"symbol": symbol.upper()}))


def get_stock_quote(symbol: str) -> dict:
    """Return a real-time quote: price, change, day range, volume.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return first_record(fmp_get("quote", {"symbol": symbol.upper()}))


def get_key_metrics(symbol: str) -> dict:
    """Return TTM key metrics: P/E, ROE, margins, debt/equity, FCF yield.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return first_record(fmp_get("key-metrics-ttm", {"symbol": symbol.upper()}))


def get_dcf_valuation(symbol: str) -> dict:
    """Return the discounted-cash-flow fair value vs current price.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return first_record(fmp_get("discounted-cash-flow", {"symbol": symbol.upper()}))


def get_analyst_ratings(symbol: str) -> dict:
    """Return analyst grade actions and the consensus price target.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    sym = symbol.upper()
    grades = fmp_get("grades", {"symbol": sym, "limit": 10})
    consensus = first_record(fmp_get("price-target-consensus", {"symbol": sym}))
    return {
        "symbol": sym,
        "recent_grades": grades,
        "price_target_consensus": consensus,
    }


def get_peers(symbol: str) -> dict:
    """Return a list of peer tickers for competitive comparison.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    data = fmp_get("stock-peers", {"symbol": symbol.upper()})
    if isinstance(data, dict):
        return data  # error dict
    return {"symbol": symbol.upper(), "peers": data}
