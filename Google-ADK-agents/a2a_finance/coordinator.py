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

remote_commodities = RemoteA2aAgent(
    name="commodities_agent",
    agent_card=_card("A2A_COMMODITIES_CARD", 8004),
    description=(
        "Remote commodities specialist (over A2A): live oil, gas, metal, coal, "
        "and other commodity prices and history via the OilPrice API."
    ),
)

# Cross-runtime (Tier B): a TypeScript agent (@openrouter/agent) exposed over A2A.
# Its card lives on the Node service (default :8100). Construction is lazy, so
# this only requires the TS server running when the coordinator routes to it.
remote_openrouter_ts = RemoteA2aAgent(
    name="openrouter_research_agent",
    agent_card=os.getenv(
        "A2A_OPENROUTER_CARD", "http://localhost:8100/.well-known/agent-card.json"
    ),
    description=(
        "Remote general equity-research agent running on a DIFFERENT runtime "
        "(TypeScript / OpenRouter Agent SDK), reachable over A2A. Good for a "
        "broad quick read or a cross-runtime second opinion."
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
        "- Broad 'give me a general/quick read on TICKER' -> transfer to "
        "openrouter_research_agent (a cross-runtime agent).\n"
        "- Commodity prices (oil, gas, metals, coal — WTI, Brent, gold, etc.) -> "
        "transfer to commodities_agent.\n"
        "Always name the ticker or commodity being analyzed."
    ),
    sub_agents=[
        remote_fundamentals,
        remote_valuation,
        remote_risk,
        remote_commodities,
        remote_openrouter_ts,
    ],
)
