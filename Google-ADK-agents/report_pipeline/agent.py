"""Root agent for the full-report pipeline.

A ``SequentialAgent`` runs the steps in order, sharing one session/state:
fundamentals -> valuation -> risk -> synthesis. The first three write their
section to state via ``output_key``; synthesis reads them back and emits the
final rated note. ADK exposes ``root_agent`` to ``adk web`` / ``adk run``.
"""

# Init tracing BEFORE importing google.adk so ADK's spans land on our provider.
from a2a_finance.observability import init_tracing

init_tracing()

from google.adk.agents import SequentialAgent

from .steps import (
    fundamentals_step,
    valuation_step,
    risk_step,
    synthesis_step,
)

root_agent = SequentialAgent(
    name="equity_report_pipeline",
    description=(
        "Generates a full equity research note for a ticker by running "
        "fundamentals, valuation, and risk analysis in sequence and synthesizing "
        "them into a rated report."
    ),
    sub_agents=[
        fundamentals_step,
        valuation_step,
        risk_step,
        synthesis_step,
    ],
)
