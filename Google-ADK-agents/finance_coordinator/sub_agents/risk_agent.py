"""Risk specialist — surfaces the key risks and red flags for a name."""

from google.adk.agents import LlmAgent

from ..config import MODEL
from ..tools import get_key_metrics, get_company_profile

risk_agent = LlmAgent(
    name="risk_agent",
    model=MODEL,
    description=(
        "Identifies the material risks for a stock: leverage, margin fragility, "
        "sector/concentration exposure, and valuation risk. Use for 'what could "
        "go wrong' questions."
    ),
    instruction=(
        "You are a risk analyst. Given a ticker, use the profile and key-metrics "
        "tools to reason about downside: balance-sheet leverage, margin "
        "durability, cyclicality of the sector, and any single-point exposures. "
        "Return a ranked list of the top 3-5 risks, each with a one-line 'why it "
        "matters'. Be specific; do not pad with generic market risk."
    ),
    tools=[get_company_profile, get_key_metrics],
)
