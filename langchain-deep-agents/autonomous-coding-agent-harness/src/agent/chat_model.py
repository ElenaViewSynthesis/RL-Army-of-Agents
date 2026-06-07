"""Chat model selection for the agent runtime."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL_PROVIDER = "google_genai"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

_GOOGLE_PROVIDERS = {"google", "google_genai", "gemini"}
_GROQ_PROVIDERS = {"groq"}


def _has_any_env(names: tuple[str, ...]) -> bool:
    return any(bool(os.environ.get(name)) for name in names)


def _google_configured() -> bool:
    return _has_any_env(("GOOGLE_API_KEY", "GEMINI_API_KEY"))


def _groq_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def _make_google_model(model: str) -> Any:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Gemini requires langchain-google-genai. Install it with "
            "`uv pip install langchain-google-genai` or run `uv sync`."
        ) from exc

    return ChatGoogleGenerativeAI(model=model)


def _make_groq_model(model: str) -> Any:
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError(
            "Groq fallback requires langchain-groq. Install it with "
            "`uv pip install langchain-groq` or run `uv sync`."
        ) from exc

    return ChatGroq(model=model)


def make_chat_model() -> Any:
    """Create the configured chat model.

    Gemini is the default provider. Groq remains available by setting
    AGENT_MODEL_PROVIDER=groq, or as an automatic fallback when Google credentials
    are absent and GROQ_API_KEY is present.
    """
    provider = os.environ.get("AGENT_MODEL_PROVIDER", DEFAULT_MODEL_PROVIDER).lower()

    if provider in _GROQ_PROVIDERS:
        model = os.environ.get("GROQ_AGENT_MODEL") or os.environ.get(
            "AGENT_MODEL",
            DEFAULT_GROQ_MODEL,
        )
        return _make_groq_model(model)

    if provider not in _GOOGLE_PROVIDERS:
        raise ValueError(
            "unsupported AGENT_MODEL_PROVIDER="
            f"{provider!r}; expected google_genai or groq"
        )

    if not _google_configured() and _groq_configured():
        model = os.environ.get("GROQ_AGENT_MODEL", DEFAULT_GROQ_MODEL)
        return _make_groq_model(model)

    model = os.environ.get("GOOGLE_GENAI_MODEL") or os.environ.get(
        "AGENT_MODEL",
        DEFAULT_GEMINI_MODEL,
    )
    return _make_google_model(model)


def get_chat_model_identifier() -> str:
    """Return the provider-qualified model string for LangChain agents."""
    provider = os.environ.get("AGENT_MODEL_PROVIDER", DEFAULT_MODEL_PROVIDER).lower()

    if provider in _GROQ_PROVIDERS:
        model = os.environ.get("GROQ_AGENT_MODEL") or os.environ.get(
            "AGENT_MODEL",
            DEFAULT_GROQ_MODEL,
        )
        return f"groq:{model}"

    if provider not in _GOOGLE_PROVIDERS:
        raise ValueError(
            "unsupported AGENT_MODEL_PROVIDER="
            f"{provider!r}; expected google_genai or groq"
        )

    if not _google_configured() and _groq_configured():
        model = os.environ.get("GROQ_AGENT_MODEL", DEFAULT_GROQ_MODEL)
        return f"groq:{model}"

    model = os.environ.get("GOOGLE_GENAI_MODEL") or os.environ.get(
        "AGENT_MODEL",
        DEFAULT_GEMINI_MODEL,
    )
    return f"google_genai:{model}"
