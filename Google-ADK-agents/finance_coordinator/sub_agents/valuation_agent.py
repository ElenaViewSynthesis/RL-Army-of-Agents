"""Valuation specialist — DCF fair value, peers, and analyst sentiment.

Runs on an **open model via OpenRouter** (LiteLLM), while the coordinator and
the other specialists run on Gemini — demonstrating heterogeneous models
coordinating inside one ADK agent tree.
"""

from google.adk.agents import LlmAgent

from ..config import openrouter_model
from ..tools import get_dcf_valuation, get_peers, get_analyst_ratings

valuation_agent = LlmAgent(
    name="valuation_agent",
    model=openrouter_model(),
    description=(
        "Assesses whether a stock is cheap or expensive: DCF fair value vs "
        "price, peer multiples, and the analyst consensus / price target. Use "
        "for 'is it worth buying at this price' questions."
    ),
    instruction=(
        "You are a valuation analyst. Given a ticker, call the DCF, peers, and "
        "analyst-ratings tools. Compare DCF fair value to the current price, "
        "note where the stock sits versus peers, and report the analyst "
        "consensus and price target. Conclude with a one-line under/over/"
        "fairly-valued read, and state your confidence given data gaps."
    ),
    tools=[get_dcf_valuation, get_peers, get_analyst_ratings],
)
