"""Tool functions exposed to the finance sub-agents.

ADK auto-wraps plain Python functions (with type hints + a docstring) as
``FunctionTool``s when they are passed in an agent's ``tools=[...]`` list.
"""

from .market_tools import (
    get_company_profile,
    get_stock_quote,
    get_key_metrics,
    get_dcf_valuation,
    get_analyst_ratings,
    get_peers,
)

__all__ = [
    "get_company_profile",
    "get_stock_quote",
    "get_key_metrics",
    "get_dcf_valuation",
    "get_analyst_ratings",
    "get_peers",
]
