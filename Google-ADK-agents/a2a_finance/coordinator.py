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

def _card(env: str, port: int) -> str:
    return os.getenv(env, f"http://localhost:{port}/.well-known/agent-card.json")


COORD_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

remote_fundamentals = RemoteA2aAgent(
    name="fundamentals_agent",
    agent_card=_card("A2A_FUNDAMENTALS_CARD", 8002),
    description=(
        "Remote fundamentals specialist (over A2A): company profile, current "
        "quote, and TTM financial health (margins, ROE, leverage, FCF)."
    ),
)

remote_valuation = RemoteA2aAgent(
    name="valuation_agent",
    agent_card=_card("A2A_VALUATION_CARD", 8001),
    description=(
        "Remote valuation specialist (over A2A): DCF fair value vs price, peer "
        "multiples, analyst consensus / price target."
    ),
)

remote_risk = RemoteA2aAgent(
    name="risk_agent",
    agent_card=_card("A2A_RISK_CARD", 8003),
    description=(
        "Remote risk specialist (over A2A): leverage, margin fragility, "
        "cyclicality, and single-point exposures."
    ),
)

root_agent = LlmAgent(
    name="finance_coordinator",
    model=OpenRouterLlm(model=COORD_MODEL),
    description="Coordinates equity-research specialists over the A2A protocol.",
    instruction=(
        "You are the lead equity-research coordinator. You do not analyze data "
        "yourself — you delegate to remote specialists and relay their findings.\n"
        "Routing:\n"
        "- Business health / financials / 'how is the company doing' -> transfer "
        "to fundamentals_agent.\n"
        "- Cheap or expensive / DCF / peers / price targets -> transfer to "
        "valuation_agent.\n"
        "- Downside / red flags / 'what could go wrong' -> transfer to "
        "risk_agent.\n"
        "For a broad 'research TICKER' request, consult the relevant specialists "
        "and combine their answers into a short brief. Always name the ticker."
    ),
    sub_agents=[remote_fundamentals, remote_valuation, remote_risk],
)
