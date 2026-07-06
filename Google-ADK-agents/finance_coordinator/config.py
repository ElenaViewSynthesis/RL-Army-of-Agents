"""Shared model configuration for the finance coordinator agents.

The tree is deliberately **multi-model**: some agents run on Gemini directly,
others on an open model served through OpenRouter (via ADK's LiteLLM wrapper).
They still coordinate as one agent tree — ADK's delegation is model-agnostic.

Env knobs (set in ``finance_coordinator/.env``):
- ``ADK_MODEL``        — Gemini model for the native agents (default 2.5-flash)
- ``OPENROUTER_MODEL`` — open model id on OpenRouter for the LiteLLM agent
- ``OPENROUTER_API_KEY`` — key for the OpenRouter-backed agent
"""

from __future__ import annotations

import os

from google.adk.models.lite_llm import LiteLlm

# ── Gemini (native ADK) ───────────────────────────────────────────────────────
# Plain model-id string; ADK routes it to the Google backend.
MODEL: str = os.getenv("ADK_MODEL", "gemini-2.5-flash")
GEMINI_MODEL: str = MODEL  # explicit alias for readability at call sites

# ── OpenRouter (open model via LiteLLM) ───────────────────────────────────────
OPENROUTER_MODEL_ID: str = os.getenv(
    "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"
)


def openrouter_model() -> LiteLlm:
    """Return a LiteLLM model handle for an open model on OpenRouter.

    litellm resolves the ``openrouter/`` prefix and authenticates with
    ``OPENROUTER_API_KEY``; we pass it explicitly so the agent fails fast with a
    clear error if the key is missing rather than at first token.
    """
    return LiteLlm(
        model=f"openrouter/{OPENROUTER_MODEL_ID}",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
