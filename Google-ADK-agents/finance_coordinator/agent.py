"""Root coordinator agent.

A multi-agent orchestrator: the coordinator holds no market tools itself. It
routes each request to the right specialist via ADK's LLM-driven delegation
(``sub_agents``). ADK exposes ``root_agent`` to ``adk web`` / ``adk run``.
"""

from google.adk.agents import LlmAgent

from .config import MODEL
from .sub_agents import fundamentals_agent, valuation_agent, risk_agent

root_agent = LlmAgent(
    name="finance_coordinator",
    model=MODEL,
    description=(
        "Coordinates equity-research specialists to answer questions about a "
        "public company."
    ),
    instruction=(
        "You are the lead equity-research coordinator. You do not analyze data "
        "yourself — you delegate to specialists and synthesize their answers.\n\n"
        "Routing:\n"
        "- Business health, financials, or 'how is the company doing' -> "
        "transfer to fundamentals_agent.\n"
        "- Whether the stock is cheap/expensive, DCF, peers, price targets -> "
        "transfer to valuation_agent.\n"
        "- Downside, red flags, 'what could go wrong' -> transfer to risk_agent.\n\n"
        "For a broad 'research TICKER' request, consult the relevant specialists "
        "and combine their findings into a short brief with a clear takeaway. "
        "Always name the ticker you are analyzing. If a specialist reports "
        "missing data, say so plainly rather than filling gaps."
    ),
    sub_agents=[fundamentals_agent, valuation_agent, risk_agent],
)
