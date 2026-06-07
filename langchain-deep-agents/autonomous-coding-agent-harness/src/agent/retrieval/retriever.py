"""Top-k tool retrieval over the registry."""

from __future__ import annotations

from agent.retrieval.embedder import EmbeddingService
from agent.retrieval.store import InMemoryVectorStore

ALWAYS_INCLUDE = frozenset({"read_file", "write_file", "git_status"})
DEFAULT_K = 8
K_WIDEN_STEP = 4


class ToolRetriever:
    """Retrieve relevant tool names for a goal."""

    def __init__(self, store: InMemoryVectorStore, embedder: EmbeddingService) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def total(self) -> int:
        return self._store.count()

    def retrieve(self, goal: str, k: int = DEFAULT_K) -> list[str]:
        query = self._embedder.embed(goal)
        hits = self._store.top_k(query, min(k, self.total))
        names = {entry.name for entry in hits}
        names |= ALWAYS_INCLUDE
        return sorted(names)

    def retrieve_wider(self, goal: str, current_k: int) -> tuple[list[str], int]:
        new_k = min(current_k + K_WIDEN_STEP, self.total)
        return self.retrieve(goal, new_k), new_k
