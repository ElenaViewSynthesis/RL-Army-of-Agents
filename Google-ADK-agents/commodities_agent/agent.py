"""Commodities agent — live commodity prices via the OilPrice API.

Runs on the OpenRouter client SDK (`OpenRouterLlm`), so no Gemini needed. ADK
discovers `root_agent`; also usable standalone via `run.py`.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from finance_coordinator.models import OpenRouterLlm
from .tools import (
    list_commodities,
    search_commodities,
    get_commodity_price,
    get_commodity_history,
)

MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

root_agent = LlmAgent(
    name="commodities_agent",
    model=OpenRouterLlm(model=MODEL),
    description=(
        "Answers questions about commodity prices (oil, gas, metals, coal, and "
        "more) using the live OilPrice API — latest prices, history, and the "
        "460+ code catalog."
    ),
    instruction=(
        "You are a commodities analyst. To answer a question:\n"
        "- If the user names a commodity but you are unsure of its code, call "
        "search_commodities to find the code first.\n"
        "- Use get_commodity_price for the latest price and get_commodity_history "
        "for trends over past_day/week/month/year.\n"
        "- Use list_commodities to browse a category (e.g. 'metal', 'gas').\n"
        "Report the price with its unit and currency (e.g. '$75.22 / barrel'), "
        "cite the code you used, and if a tool reports missing data say so "
        "rather than inventing figures."
    ),
    tools=[list_commodities, search_commodities, get_commodity_price, get_commodity_history],
)
