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

# ── OpenRouter (open reasoning model via LiteLLM) ─────────────────────────────
# Default to a nemotron reasoning model. Reasoning is enabled per OpenRouter's
# `reasoning` body param, forwarded through LiteLLM via `extra_body`.
OPENROUTER_MODEL_ID: str = os.getenv(
    "OPENROUTER_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
)
# Toggle reasoning off with OPENROUTER_REASONING=0/false for non-reasoning models.
OPENROUTER_REASONING: bool = os.getenv("OPENROUTER_REASONING", "1").lower() not in (
    "0",
    "false",
    "no",
)


def openrouter_model(
    model_id: str | None = None, reasoning: bool | None = None
) -> LiteLlm:
    """Return a LiteLLM model handle for an open model on OpenRouter.

    litellm resolves the ``openrouter/`` prefix and authenticates with
    ``OPENROUTER_API_KEY``; we pass it explicitly so the agent fails fast with a
    clear error if the key is missing rather than at first token.

    When reasoning is enabled we forward ``extra_body={"reasoning": {"enabled":
    True}}`` — OpenRouter's flag to make the model think before it answers. ADK
    manages the multi-turn message loop, so reasoning traces are threaded across
    turns by the framework rather than by hand-passing ``reasoning_details``.

    Args:
        model_id: Override the OpenRouter model id (defaults to
            ``OPENROUTER_MODEL_ID``).
        reasoning: Override the reasoning toggle. Set ``False`` for structured
            (JSON) output, where a reasoning trace would corrupt the payload.
    """
    mid = model_id or OPENROUTER_MODEL_ID
    use_reasoning = OPENROUTER_REASONING if reasoning is None else reasoning
    extra = {"reasoning": {"enabled": True}} if use_reasoning else {}
    return LiteLlm(
        model=f"openrouter/{mid}",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        extra_body=extra,
    )


# Model for structured/JSON output (formatter agent). Reasoning off so the trace
# doesn't pollute the JSON; instruction-following model by default.
FORMATTER_MODEL_ID: str = os.getenv(
    "OPENROUTER_FORMATTER_MODEL", "meta-llama/llama-3.3-70b-instruct"
)


def formatter_model() -> LiteLlm:
    """A non-reasoning OpenRouter model for schema-constrained output."""
    return openrouter_model(model_id=FORMATTER_MODEL_ID, reasoning=False)
