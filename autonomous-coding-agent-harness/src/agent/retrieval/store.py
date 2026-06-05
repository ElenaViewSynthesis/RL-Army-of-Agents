"""Vector store abstractions for retrieval."""

from __future__ import annotations

import math

from agent.retrieval.embedder import Vector
from agent.retrieval.registry import ToolRegistryEntry


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
