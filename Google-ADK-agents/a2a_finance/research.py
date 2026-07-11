"""Fan-out research agent — consults all specialists, synthesizes one note.

Unlike `coordinator.py` (which *routes* one query to one specialist via
`transfer_to_agent`, handing off control), this agent wraps each remote
specialist as an `AgentTool`. Calling a tool **returns control**, so the model
can consult all three A2A specialists in one turn and then synthesize a single
rated research note.

    fundamentals_agent  ─┐
    valuation_agent      ├─ called as AgentTools (over A2A) → control returns
    risk_agent          ─┘
             ▼
    research_agent synthesizes → one note (rating + target)

Needs the three Python A2A services running (:8001/:8002/:8003).
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool

from finance_coordinator.models import OpenRouterLlm


def _card(env: str, port: int) -> str:
    return os.getenv(env, f"http://localhost:{port}/.well-known/agent-card.json")


# Fresh remote handles (kept separate from coordinator.py's sub_agents so there's
# no single-parent conflict); wrapped as tools rather than sub-agents.
_fundamentals = RemoteA2aAgent(
    name="fundamentals_agent",
    agent_card=_card("A2A_FUNDAMENTALS_CARD", 8002),
    description="Remote fundamentals specialist (profile, quote, TTM metrics).",
)
_valuation = RemoteA2aAgent(
    name="valuation_agent",
    agent_card=_card("A2A_VALUATION_CARD", 8001),
    description="Remote valuation specialist (DCF, peers, analyst consensus).",
)
_risk = RemoteA2aAgent(
    name="risk_agent",
    agent_card=_card("A2A_RISK_CARD", 8003),
    description="Remote risk specialist (leverage, margins, red flags).",
)

MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

root_agent = LlmAgent(
    name="research_agent",
    model=OpenRouterLlm(model=MODEL),
    description="Produces a full research note by consulting all A2A specialists.",
    instruction=(
        "You are the lead analyst. For a research request on a ticker, gather "
        "all three perspectives by calling EACH specialist tool exactly once "
        "with the ticker: fundamentals_agent, valuation_agent, and risk_agent. "
        "Then synthesize their outputs into ONE research note with this shape:\n\n"
        "# <TICKER> — Research Note\n"
        "**Rating:** BUY / HOLD / SELL  ·  **12-month target:** <value or n/a>\n\n"
        "## Executive Summary\n(3-4 sentences tying the three views together.)\n\n"
        "## Fundamentals\n(from fundamentals_agent)\n\n"
        "## Valuation\n(from valuation_agent)\n\n"
        "## Risks\n(from risk_agent)\n\n"
        "Derive the rating/target from the valuation and risk findings. Do not "
        "invent data the specialists did not provide; note gaps plainly."
    ),
    tools=[
        AgentTool(agent=_fundamentals),
        AgentTool(agent=_valuation),
        AgentTool(agent=_risk),
    ],
)
