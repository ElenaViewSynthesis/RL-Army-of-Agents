"""Vector store abstractions for retrieval."""

from __future__ import annotations

import json
import math
import uuid
from typing import Any

from agent.retrieval.embedder import Embedder, Vector
from agent.retrieval.registry import ToolRegistryEntry, entry_text

DEFAULT_PGVECTOR_TABLE = "tool_registry_vectors"


def _cosine(left: Vector, right: Vector) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class InMemoryVectorStore:
    """Small local vector store used before the pgvector slice."""

    def __init__(self) -> None:
        self._rows: list[tuple[ToolRegistryEntry, Vector]] = []

    def upsert(self, entries: list[ToolRegistryEntry], embeddings: list[Vector]) -> None:
        by_tool = {
            (entry.namespace, entry.name): (entry, vector)
            for entry, vector in self._rows
        }
        for entry, vector in zip(entries, embeddings):
            by_tool[(entry.namespace, entry.name)] = (entry, vector)
        self._rows = list(by_tool.values())

    def top_k(self, query: Vector, k: int) -> list[ToolRegistryEntry]:
        scored = [
            (_cosine(query, vector), entry)
            for entry, vector in self._rows
        ]
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [entry for _, entry in scored[:k]]

    def count(self) -> int:
        return len(self._rows)


class PgVectorStore:
    """LangChain PGVectorStore adapter for production retrieval."""

    def __init__(
        self,
        dsn: str,
        embedding_service: Embedder,
        *,
        table_name: str = DEFAULT_PGVECTOR_TABLE,
        vector_size: int | None = None,
    ) -> None:
        self._dsn = _normalize_pgvector_url(dsn)
        self._embedding_service = embedding_service
        self._table_name = table_name
        self._vector_size = vector_size or embedding_service.dimensions
        self._engine: Any | None = None
        self._store: Any | None = None
        self._count = 0

    def init_schema(self) -> None:
        """Initialize the LangChain pgvector table and store adapter."""
        from langchain_postgres import PGEngine
        from langchain_postgres import PGVectorStore as LangChainPGVectorStore

        self._engine = PGEngine.from_connection_string(url=self._dsn)
        try:
            self._engine.init_vectorstore_table(
                table_name=self._table_name,
                vector_size=self._vector_size,
            )
        except Exception as exc:
            if not _is_duplicate_table_error(exc):
                raise
        self._store = LangChainPGVectorStore.create_sync(
            engine=self._engine,
            table_name=self._table_name,
            embedding_service=self._embedding_service,
        )

    def upsert(self, entries: list[ToolRegistryEntry], _embeddings: list[Vector]) -> None:
        if self._store is None:
            raise RuntimeError(
                "PgVectorStore.init_schema() must be called before upsert()."
            )

        ids = [_entry_id(entry) for entry in entries]
        self._store.delete(ids=ids)
        self._store.add_texts(
            texts=[entry_text(entry) for entry in entries],
            metadatas=[_metadata(entry) for entry in entries],
            ids=ids,
        )
        self._count = len(ids)

    def top_k(self, query: Vector, k: int) -> list[ToolRegistryEntry]:
        if self._store is None:
            raise RuntimeError(
                "PgVectorStore.init_schema() must be called before top_k()."
            )

        docs = self._store.similarity_search_by_vector(query, k=k)
        return [_entry_from_metadata(doc.metadata, doc.page_content) for doc in docs]

    def count(self) -> int:
        return self._count


def _schema_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _entry_id(entry: ToolRegistryEntry) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{entry.namespace}:{entry.name}"))


def _metadata(entry: ToolRegistryEntry) -> dict[str, Any]:
    return {
        "namespace": entry.namespace,
        "name": entry.name,
        "description": entry.description,
        "input_schema": entry.input_schema,
    }


def _entry_from_metadata(metadata: dict[str, Any], page_content: str) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        namespace=str(metadata.get("namespace", "")),
        name=str(metadata.get("name", "")),
        description=str(metadata.get("description") or page_content),
        input_schema=_schema_dict(metadata.get("input_schema", {})),
    )


def _normalize_pgvector_url(dsn: str) -> str:
    if dsn.startswith("postgresql+"):
        return dsn
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+asyncpg://", 1)
    return dsn


def _is_duplicate_table_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "duplicate" in message
        or "already exists" in message
        or "42p07" in message
    )
