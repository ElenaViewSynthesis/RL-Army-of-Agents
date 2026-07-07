"""Coordinator that delegates to remote agents over A2A.

The valuation specialist is not an in-process `sub_agent` here — it's a
`RemoteA2aAgent` pointing at the A2A service's agent card. ADK routes the
coordinator's delegation to it over HTTP. The coordinator's own model is the
OpenRouter client SDK (`OpenRouterLlm`), so the whole flow runs without Gemini.

Add future specialists the same way: stand up another `to_a2a` service and
register another `RemoteA2aAgent` in `sub_agents`.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

from finance_coordinator.models import OpenRouterLlm

VALUATION_CARD = os.getenv(
    "A2A_VALUATION_CARD", "http://localhost:8001/.well-known/agent-card.json"
)
COORD_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

remote_valuation = RemoteA2aAgent(
    name="valuation_agent",
    agent_card=VALUATION_CARD,
    description=(
        "Remote valuation specialist (over A2A): DCF fair value vs price, peer "
        "multiples, analyst consensus / price target."
    ),
)

root_agent = LlmAgent(
    name="finance_coordinator",
    model=OpenRouterLlm(model=COORD_MODEL),
    description="Coordinates equity-research specialists over the A2A protocol.",
    instruction=(
        "You are the lead equity-research coordinator. You do not analyze data "
        "yourself. For any valuation question — whether a stock is cheap or "
        "expensive, DCF, peers, or price targets — transfer to valuation_agent, "
        "then relay its findings. Always name the ticker being analyzed."
    ),
    sub_agents=[remote_valuation],
)
