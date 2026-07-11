"""Commodities specialist as an A2A **server**.

Wraps the existing `commodities_agent` (OilPrice API tools, OpenRouterLlm) and
exposes it over A2A so the coordinator can route commodity-price questions to it.
Serve with:

    uv run uvicorn a2a_finance.commodities_service:a2a_app --port 8004

Agent card: http://localhost:8004/.well-known/agent-card.json
"""

from __future__ import annotations

import os

from google.adk.a2a.utils.agent_to_a2a import to_a2a

from commodities_agent.agent import root_agent as commodities_agent

PORT = int(os.getenv("A2A_COMMODITIES_PORT", "8004"))

a2a_app = to_a2a(commodities_agent, port=PORT)
