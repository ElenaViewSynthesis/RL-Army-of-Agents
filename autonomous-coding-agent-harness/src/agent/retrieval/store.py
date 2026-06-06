"""Vector store abstractions for retrieval."""

from __future__ import annotations

import json
import math
from typing import Any

from agent.retrieval.embedder import Vector
from agent.retrieval.registry import ToolRegistryEntry

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tool_registry (
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    input_schema JSONB NOT NULL,
    embedding vector(128) NOT NULL,
    PRIMARY KEY (namespace, name)
);

CREATE INDEX IF NOT EXISTS tool_registry_embedding_hnsw
    ON tool_registry USING hnsw (embedding vector_cosine_ops);
"""

_UPSERT_SQL = """
INSERT INTO tool_registry (namespace, name, description, input_schema, embedding)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (namespace, name) DO UPDATE SET
    description = EXCLUDED.description,
    input_schema = EXCLUDED.input_schema,
    embedding = EXCLUDED.embedding;
"""

_TOP_K_SQL = """
SELECT namespace, name, description, input_schema
FROM tool_registry
ORDER BY embedding <=> %s
LIMIT %s;
"""


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
        by_name = {entry.name: (entry, vector) for entry, vector in self._rows}
        for entry, vector in zip(entries, embeddings):
            by_name[entry.name] = (entry, vector)
        self._rows = list(by_name.values())

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
    """PostgreSQL + pgvector store for production retrieval."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def init_schema(self) -> None:
        """Create the extension, table, and vector index if needed."""
        with self._connect() as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()

    def upsert(self, entries: list[ToolRegistryEntry], embeddings: list[Vector]) -> None:
        rows = [
            (
                entry.namespace,
                entry.name,
                entry.description,
                json.dumps(entry.input_schema),
                embedding,
            )
            for entry, embedding in zip(entries, embeddings)
        ]
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(_UPSERT_SQL, rows)
            conn.commit()

    def top_k(self, query: Vector, k: int) -> list[ToolRegistryEntry]:
        with self._connect() as conn:
            rows = conn.execute(_TOP_K_SQL, (query, k)).fetchall()
        return [
            ToolRegistryEntry(
                namespace=row[0],
                name=row[1],
                description=row[2],
                input_schema=_schema_dict(row[3]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM tool_registry;").fetchone()
        return int(row[0]) if row else 0


def _schema_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}
