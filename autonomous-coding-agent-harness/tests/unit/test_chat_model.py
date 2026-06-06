from __future__ import annotations

from types import SimpleNamespace

from agent.chat_model import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    make_chat_model,
)


class FakeGemini:
    def __init__(self, model: str) -> None:
        self.model = model


class FakeGroq:
    def __init__(self, model: str) -> None:
        self.model = model


def _install_fake_models(monkeypatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=FakeGemini),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "langchain_groq",
        SimpleNamespace(ChatGroq=FakeGroq),
    )


def test_default_chat_model_uses_gemini(monkeypatch) -> None:
    _install_fake_models(monkeypatch)
    monkeypatch.delenv("AGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_MODEL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    model = make_chat_model()

    assert isinstance(model, FakeGemini)
    assert model.model == DEFAULT_GEMINI_MODEL


def test_groq_provider_uses_secondary_model(monkeypatch) -> None:
    _install_fake_models(monkeypatch)
    monkeypatch.setenv("AGENT_MODEL_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_AGENT_MODEL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)

    model = make_chat_model()

    assert isinstance(model, FakeGroq)
    assert model.model == DEFAULT_GROQ_MODEL


def test_groq_fallback_when_google_key_missing(monkeypatch) -> None:
    _install_fake_models(monkeypatch)
    monkeypatch.delenv("AGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    model = make_chat_model()

    assert isinstance(model, FakeGroq)
    assert model.model == DEFAULT_GROQ_MODEL
