"""Energy Drilling agent — EIA Drilling Productivity Report (DPR) via OilPrice.

DUC (Drilled but Uncompleted) well inventories and per-rig oil/gas productivity by
basin. Runs on the OpenRouter client SDK (`OpenRouterLlm`) — no Gemini needed.
ADK discovers `root_agent`; also usable standalone via `run.py`.

**The DPR endpoints are PREMIUM** (OilPrice "Scale" plan). They're wired in full so
anyone with a Scale key can run them; on the free tier the tools return a clear
"premium endpoint" error and the agent reports the gap.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from finance_coordinator.models import OpenRouterLlm
from .tools import (
    list_drilling_reports,
    get_latest_drilling_report,
    get_drilling_summary,
    get_duc_wells,
    get_drilling_by_basin,
    get_drilling_historical,
    get_drilling_trends,
    get_drilling_report,
)

MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

root_agent = LlmAgent(
    name="energy_drilling_agent",
    model=OpenRouterLlm(model=MODEL),
    description=(
        "Answers questions on US shale drilling productivity and DUC (Drilled but "
        "Uncompleted) well inventories from the EIA Drilling Productivity Report, "
        "by basin (Permian, Bakken, Eagle Ford, Niobrara, Appalachia, Anadarko, "
        "Haynesville). Premium OilPrice 'Scale' endpoint."
    ),
    instruction=(
        "You are an upstream-oil-&-gas analyst covering US shale drilling.\n"
        "- For the current picture use get_latest_drilling_report or "
        "get_drilling_summary; for DUC counts + month-over-month use get_duc_wells.\n"
        "- For one basin over time use get_drilling_historical; for several basins "
        "use get_drilling_by_basin; for cross-basin direction use get_drilling_trends.\n"
        "- Valid basins: permian, bakken, eagle_ford, niobrara, appalachia, "
        "anadarko, haynesville.\n"
        "Interpretation: DUCs are drilled-but-uncompleted wells — a rising DUC count "
        "means deferred completions (ready supply), a falling count often precedes "
        "production increases. Report productivity with units (bbl/d per rig, mcf/d "
        "per rig).\n"
        "IMPORTANT: these are PREMIUM endpoints. If a tool returns an error saying a "
        "'Scale' plan is required, tell the user this dataset needs OilPrice's Scale "
        "tier rather than inventing figures."
    ),
    tools=[
        list_drilling_reports,
        get_latest_drilling_report,
        get_drilling_summary,
        get_duc_wells,
        get_drilling_by_basin,
        get_drilling_historical,
        get_drilling_trends,
        get_drilling_report,
    ],
)
