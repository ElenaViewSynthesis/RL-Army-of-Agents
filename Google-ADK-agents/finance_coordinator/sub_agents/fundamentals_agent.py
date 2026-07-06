"""Fundamentals specialist — company profile and TTM financial health."""

from google.adk.agents import LlmAgent

from ..config import MODEL
from ..tools import get_company_profile, get_stock_quote, get_key_metrics

fundamentals_agent = LlmAgent(
    name="fundamentals_agent",
    model=MODEL,
    description=(
        "Analyzes a company's fundamentals: profile, current quote, and TTM "
        "metrics (margins, ROE, leverage, FCF yield). Use for 'how is the "
        "business doing' questions."
    ),
    instruction=(
        "You are an equity fundamentals analyst. Given a ticker, call the "
        "profile, quote, and key-metrics tools, then summarize the company's "
        "financial health in a few tight bullets: what it does, current price "
        "context, profitability, and balance-sheet strength. Flag any metric "
        "that is missing rather than inventing it."
    ),
    tools=[get_company_profile, get_stock_quote, get_key_metrics],
)
