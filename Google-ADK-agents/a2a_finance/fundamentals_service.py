"""Fundamentals specialist as an A2A **server**.

Model = OpenRouter client SDK (`OpenRouterLlm`). Serve with:

    uv run uvicorn a2a_finance.fundamentals_service:a2a_app --port 8002

Agent card: http://localhost:8002/.well-known/agent-card.json
"""

from __future__ import annotations

import os

# Init tracing BEFORE importing google.adk so ADK's spans land on our provider.
from a2a_finance.observability import init_tracing

init_tracing()

from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from finance_coordinator.models import OpenRouterLlm
from finance_coordinator.tools import get_company_profile, get_stock_quote, get_key_metrics

PORT = int(os.getenv("A2A_FUNDAMENTALS_PORT", "8002"))
# Fast, reliable tool-calling model — the reasoning models on OpenRouter's free
# tier queue for minutes, which is the fan-out bottleneck. Override via env.
MODEL = os.getenv("A2A_FUNDAMENTALS_MODEL", "meta-llama/llama-3.3-70b-instruct")

fundamentals_agent = LlmAgent(
    name="fundamentals_agent",
    model=OpenRouterLlm(model=MODEL),
    description=(
        "Analyzes a company's fundamentals: profile, current quote, and TTM "
        "metrics (margins, ROE, leverage, FCF yield)."
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

a2a_app = to_a2a(fundamentals_agent, port=PORT)
