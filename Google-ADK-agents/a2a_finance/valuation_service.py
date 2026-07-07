"""Valuation specialist as an A2A **server**.

Runs the valuation agent (model = OpenRouter client SDK via `OpenRouterLlm`)
and exposes it over the A2A protocol with `to_a2a`. Serve it with:

    uv run uvicorn a2a_finance.valuation_service:a2a_app --port 8001

The coordinator then reaches it via its agent card at
http://localhost:8001/.well-known/agent-card.json (see coordinator.py).
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from finance_coordinator.models import OpenRouterLlm
from finance_coordinator.tools import get_dcf_valuation, get_peers, get_analyst_ratings

PORT = int(os.getenv("A2A_VALUATION_PORT", "8001"))
# Reliable tool-calling model for the service. Override with A2A_VALUATION_MODEL.
MODEL = os.getenv("A2A_VALUATION_MODEL", "meta-llama/llama-3.3-70b-instruct")

valuation_agent = LlmAgent(
    name="valuation_agent",
    model=OpenRouterLlm(model=MODEL),
    description=(
        "Assesses whether a stock is cheap or expensive: DCF fair value vs "
        "price, peer multiples, and analyst consensus / price target."
    ),
    instruction=(
        "You are a valuation analyst. Given a ticker, call the DCF, peers, and "
        "analyst-ratings tools, then report: DCF fair value vs current price, "
        "where it sits versus peers, and the analyst consensus / target. End "
        "with a one-line under/over/fairly-valued read and your confidence."
    ),
    tools=[get_dcf_valuation, get_peers, get_analyst_ratings],
)

# ASGI app exposing the agent over A2A (agent card advertises this port).
a2a_app = to_a2a(valuation_agent, port=PORT)
