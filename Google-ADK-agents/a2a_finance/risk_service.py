"""Risk specialist as an A2A **server**.

Model = OpenRouter client SDK (`OpenRouterLlm`). Serve with:

    uv run uvicorn a2a_finance.risk_service:a2a_app --port 8003

Agent card: http://localhost:8003/.well-known/agent-card.json
"""

from __future__ import annotations

import os

# Init tracing BEFORE importing google.adk so ADK's spans land on our provider.
from a2a_finance.observability import init_tracing

init_tracing()

from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from finance_coordinator.models import OpenRouterLlm
from finance_coordinator.tools import get_company_profile, get_key_metrics

PORT = int(os.getenv("A2A_RISK_PORT", "8003"))
MODEL = os.getenv("A2A_RISK_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")

risk_agent = LlmAgent(
    name="risk_agent",
    model=OpenRouterLlm(model=MODEL),
    description=(
        "Identifies the material risks for a stock: leverage, margin fragility, "
        "sector/concentration exposure, and valuation risk."
    ),
    instruction=(
        "You are a risk analyst. There is NO 'risks' tool — you DERIVE the risks "
        "yourself by reasoning over data. First call get_company_profile and "
        "get_key_metrics, then infer downside from the numbers: high debt/equity "
        "-> leverage risk; thin or falling margins -> margin fragility; the "
        "sector/industry in the profile -> cyclicality or regulatory exposure; a "
        "single dominant product line -> concentration risk. Return a ranked list "
        "of the top 3-5 risks, each with a one-line 'why it matters' grounded in "
        "the specific figures you pulled. Never reply that you cannot assess risk "
        "— always produce the list from the available data.\n"
        "Output ONLY the ranked risk list (one risk per line). Do NOT echo or "
        "repeat the raw tool data, and do not list metrics that are not risks."
    ),
    tools=[get_company_profile, get_key_metrics],
)

a2a_app = to_a2a(risk_agent, port=PORT)
