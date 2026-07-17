"""Graceful-degradation tests for the A2A persistence + recall layer.

These assert the core promise: when Supabase isn't configured and/or the local
embedding model isn't installed, every persistence/recall entry point **no-ops
without raising** and makes no network call. Pure Python — no DB, network, or
model required (all external effects are monkeypatched or unreachable).
"""

from __future__ import annotations

import pytest

from a2a_finance import embeddings, observability, storage


# ── storage: not configured (no SUPABASE_* env) ───────────────────────────────
@pytest.fixture
def disabled_storage(monkeypatch, tmp_path):
    """Force storage into a deterministic 'not configured' state.

    Point the .env autoloader at a nonexistent file so it can't pick up real
    credentials, and clear the relevant env + module globals.
    """
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("A2A_RUN_ID", raising=False)
    monkeypatch.setattr(storage, "_ENV_PATH", tmp_path / "does-not-exist.env")
    monkeypatch.setattr(storage, "_env_loaded", False)
    monkeypatch.setattr(storage, "_disabled", False)
    monkeypatch.setattr(storage, "_process_run_id", None)
    return storage


def test_enabled_false_without_config(disabled_storage):
    assert disabled_storage.enabled() is False


def test_writes_are_noops_when_disabled(disabled_storage):
    # None of these may raise or attempt a network call.
    assert disabled_storage.start_run("agent", subject="X", prompt="p") is None
    disabled_storage.save_price("WTI", 70.0, currency="USD", source="oilprice")
    disabled_storage.save_response("agent", subject="X", text="note", rating="HOLD")
    assert disabled_storage.current_run_id() is None


def test_search_similar_empty_when_disabled(disabled_storage):
    assert disabled_storage.search_similar("is NVDA cheap", subject="NVDA") == []


def test_current_run_id_falls_back_to_env(monkeypatch):
    # ContextVar + process global empty -> A2A_RUN_ID env is the last fallback
    # (the cross-process propagation path used by run_demo's subprocesses).
    monkeypatch.setattr(storage, "_process_run_id", None)
    storage._run_id_var.set(None)
    monkeypatch.setenv("A2A_RUN_ID", "env-run-42")
    assert storage.current_run_id() == "env-run-42"


# ── embeddings: model unavailable (recall extra not installed) ────────────────
@pytest.fixture
def no_model(monkeypatch):
    """Force the embedding model into the 'unavailable' state deterministically."""
    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "_model_failed", True)
    return embeddings


def test_embeddings_unavailable(no_model):
    assert no_model.available() is False


def test_embed_returns_none_without_model(no_model):
    assert no_model.embed("some note text") is None
    assert no_model.embed("a query", is_query=True) is None


def test_embed_empty_text_returns_none(no_model):
    assert no_model.embed("") is None


def test_to_pgvector_format():
    assert embeddings.to_pgvector([0.1, -0.2, 0.3]) == "[0.1,-0.2,0.3]"
    assert embeddings.to_pgvector([]) == "[]"


def test_embedding_dim_is_1536():
    assert embeddings.EMBEDDING_DIM == 1536


# ── save_response: embed-on-write is opt-in (default off) ─────────────────────
@pytest.fixture
def enabled_storage(monkeypatch):
    """Enable storage with a stubbed HTTP layer so nothing leaves the process."""
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test")
    monkeypatch.setattr(storage, "_env_loaded", True)  # skip the .env autoloader
    monkeypatch.setattr(storage, "_disabled", False)
    posted: dict = {}
    monkeypatch.setattr(storage, "_post",
                        lambda table, row, **kw: posted.update(table=table, row=row))
    return storage, posted


def test_save_response_skips_embedding_by_default(enabled_storage, monkeypatch):
    store, posted = enabled_storage
    monkeypatch.delenv("A2A_EMBED_ON_WRITE", raising=False)

    called = {"embed": False}

    def spy_embed(*a, **k):
        called["embed"] = True
        return None

    monkeypatch.setattr(embeddings, "embed", spy_embed)

    store.save_response("agent", subject="NVDA", text="note", run_id="rid-1")
    assert posted["table"] == "agent_responses"
    assert "embedding" not in posted["row"]   # default path: not embedded
    assert called["embed"] is False           # heavy model never touched on write


def test_save_response_embeds_when_opted_in(enabled_storage, monkeypatch):
    store, posted = enabled_storage
    monkeypatch.setenv("A2A_EMBED_ON_WRITE", "1")
    monkeypatch.setattr(embeddings, "embed", lambda text, is_query=False: [0.5, 0.25])
    monkeypatch.setattr(embeddings, "to_pgvector", lambda v: "[0.5,0.25]")

    store.save_response("agent", subject="NVDA", text="note", run_id="rid-1")
    assert posted["row"]["embedding"] == "[0.5,0.25]"


# ── observability: no-op without Langfuse credentials ────────────────────────
def test_observability_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(observability, "_initialized", False)
    assert observability.configured() is False
    assert observability.init_tracing() is False   # no-op; never imports langfuse
    observability.flush()                          # no-op, must not raise
