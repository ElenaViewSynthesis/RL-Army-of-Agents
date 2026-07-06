"""Pipeline steps for the sequential full-report generator.

Each step is a fresh ``LlmAgent`` (an agent instance can have only one parent,
so these are distinct from the coordinator's sub-agents). Each analytical step
writes its section to session state via ``output_key``; the final synthesis
step reads them back through ``{state_key}`` instruction templating.

Tools are reused from ``finance_coordinator`` so the two agents stay in sync.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from finance_coordinator.config import MODEL
from finance_coordinator.tools import (
    get_company_profile,
    get_stock_quote,
    get_key_metrics,
    get_dcf_valuation,
    get_analyst_ratings,
    get_peers,
)

# 1 ── Fundamentals ───────────────────────────────────────────────────────────
fundamentals_step = LlmAgent(
    name="fundamentals_step",
    model=MODEL,
    description="Writes the fundamentals section of the report.",
    instruction=(
        "Identify the ticker in the user's request. Call the profile, quote, and "
        "key-metrics tools for it, then write the **Fundamentals** section: what "
        "the company does, current price context, profitability (margins, ROE), "
        "and balance-sheet strength (leverage, FCF). Use tight markdown bullets. "
        "State any metric the tools returned as missing rather than inventing it."
    ),
    tools=[get_company_profile, get_stock_quote, get_key_metrics],
    output_key="fundamentals_section",
)

# 2 ── Valuation ──────────────────────────────────────────────────────────────
valuation_step = LlmAgent(
    name="valuation_step",
    model=MODEL,
    description="Writes the valuation section of the report.",
    instruction=(
        "For the same ticker, call the DCF, peers, and analyst-ratings tools. "
        "Write the **Valuation** section: DCF fair value vs current price, where "
        "the stock sits versus peers, and the analyst consensus / price target. "
        "End with a one-line under/over/fairly-valued read and your confidence "
        "given any data gaps."
    ),
    tools=[get_dcf_valuation, get_peers, get_analyst_ratings],
    output_key="valuation_section",
)

# 3 ── Risk ───────────────────────────────────────────────────────────────────
risk_step = LlmAgent(
    name="risk_step",
    model=MODEL,
    description="Writes the risk section of the report.",
    instruction=(
        "For the same ticker, use the profile and key-metrics tools to reason "
        "about downside. Write the **Risks** section as a ranked list of the top "
        "3-5 material risks (leverage, margin fragility, cyclicality, single-point "
        "exposures), each with a one-line 'why it matters'. Be specific; no generic "
        "boilerplate."
    ),
    tools=[get_company_profile, get_key_metrics],
    output_key="risk_section",
)

# 4 ── Synthesis (no tools; reads the three sections from state) ───────────────
synthesis_step = LlmAgent(
    name="synthesis_step",
    model=MODEL,
    description="Synthesizes the sections into a final rated report.",
    instruction=(
        "You are the lead analyst. Using ONLY the three sections below, write a "
        "concise institutional research note in markdown with this structure:\n\n"
        "# <TICKER> — Research Note\n"
        "**Rating:** BUY / HOLD / SELL  ·  **12-month price target:** <value or "
        "'n/a — insufficient data'>\n\n"
        "## Executive Summary\n(3-4 sentences tying the sections together.)\n\n"
        "## Fundamentals\n{fundamentals_section}\n\n"
        "## Valuation\n{valuation_section}\n\n"
        "## Risks\n{risk_section}\n\n"
        "Derive the rating and target from the valuation and risk material. If the "
        "data was incomplete, say so plainly and lower your stated confidence "
        "rather than fabricating numbers."
    ),
    # Rely on the templated sections, not the noisy raw tool history.
    include_contents="none",
    output_key="final_report",
)
