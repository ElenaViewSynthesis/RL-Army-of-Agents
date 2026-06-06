import uuid

from agent.retrieval.registry import ToolRegistryEntry
from agent.retrieval.store import (
    _entry_from_metadata,
    _entry_id,
    _metadata,
    _normalize_pgvector_url,
    _schema_dict,
)


def test_schema_dict_accepts_dict() -> None:
    assert _schema_dict({"path": {"type": "string"}}) == {"path": {"type": "string"}}


def test_schema_dict_parses_json_string() -> None:
    assert _schema_dict('{"path": {"type": "string"}}') == {"path": {"type": "string"}}


def test_pgvector_metadata_round_trips_registry_entry() -> None:
    entry = ToolRegistryEntry(
        namespace="fs",
        name="read_file",
        description="Read a file",
        input_schema={"path": {"type": "string"}},
    )

    restored = _entry_from_metadata(_metadata(entry), "fallback content")

    assert restored == entry


def test_pgvector_entry_id_is_stable_uuid() -> None:
    entry = ToolRegistryEntry(
        namespace="git",
        name="git_status",
        description="Show status",
        input_schema={},
    )

    assert _entry_id(entry) == _entry_id(entry)
    assert uuid.UUID(_entry_id(entry))


def test_normalize_pgvector_url_uses_async_driver() -> None:
    assert (
        _normalize_pgvector_url("postgresql://user:pass@localhost/db")
        == "postgresql+asyncpg://user:pass@localhost/db"
    )
    assert (
        _normalize_pgvector_url("postgres://user:pass@localhost/db")
        == "postgresql+asyncpg://user:pass@localhost/db"
    )
    assert (
        _normalize_pgvector_url("postgresql+psycopg://user:pass@localhost/db")
        == "postgresql+psycopg://user:pass@localhost/db"
    )
