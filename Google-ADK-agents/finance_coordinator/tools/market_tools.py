"""Market-data tools for the finance sub-agents.

These are intentionally stubbed with representative shapes so the agent tree
runs end-to-end without a live data provider. Swap the bodies for real
Financial Modeling Prep (FMP) `/stable` calls when wiring in the API key —
the function signatures and return dicts already match the FMP field names
used by the sibling `Equity-Research-agent`.

Each function returns a JSON-serializable dict. ADK wraps them as tools
automatically; the docstring becomes the tool description the model sees, so
keep the first line crisp and action-oriented.
"""

from __future__ import annotations


def get_company_profile(symbol: str) -> dict:
    """Return company profile: name, sector, industry, market cap, description.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return {
        "symbol": symbol.upper(),
        "companyName": f"{symbol.upper()} (stub)",
        "sector": "Unknown",
        "industry": "Unknown",
        "marketCap": None,
        "description": "Stubbed profile — wire up FMP /stable/profile.",
    }


def get_stock_quote(symbol: str) -> dict:
    """Return a real-time quote: price, change, day range, volume.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return {
        "symbol": symbol.upper(),
        "price": None,
        "changePercentage": None,
        "dayLow": None,
        "dayHigh": None,
        "volume": None,
        "_note": "Stubbed — wire up FMP /stable/quote.",
    }


def get_key_metrics(symbol: str) -> dict:
    """Return TTM key metrics: P/E, ROE, margins, debt/equity, FCF yield.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return {
        "symbol": symbol.upper(),
        "peRatioTTM": None,
        "roeTTM": None,
        "netProfitMarginTTM": None,
        "debtToEquityTTM": None,
        "freeCashFlowYieldTTM": None,
        "_note": "Stubbed — wire up FMP /stable/key-metrics-ttm.",
    }


def get_dcf_valuation(symbol: str) -> dict:
    """Return the discounted-cash-flow fair value vs current price.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return {
        "symbol": symbol.upper(),
        "dcf": None,
        "stockPrice": None,
        "_note": "Stubbed — wire up FMP /stable/discounted-cash-flow.",
    }


def get_analyst_ratings(symbol: str) -> dict:
    """Return analyst grade distribution and consensus price target.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return {
        "symbol": symbol.upper(),
        "consensus": None,
        "priceTargetConsensus": None,
        "_note": "Stubbed — wire up FMP /stable/grades + price-target-consensus.",
    }


def get_peers(symbol: str) -> dict:
    """Return a list of peer tickers for competitive comparison.

    Args:
        symbol: Stock ticker, e.g. "NVDA".
    """
    return {
        "symbol": symbol.upper(),
        "peers": [],
        "_note": "Stubbed — wire up FMP /stable/stock-peers.",
    }
