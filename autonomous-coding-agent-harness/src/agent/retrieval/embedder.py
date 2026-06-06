"""Deterministic local text embedder for tool retrieval.

The production target is sentence-transformer embeddings stored in pgvector.
This milestone starts with a small hashing embedder so retrieval behavior is
testable without downloading a model or running a database.
"""

from __future__ import annotations

import hashlib
import math
import re

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:  # type: ignore[no-redef]
        """Fallback base for local retrieval tests without LangChain installed."""


Vector = list[float]

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class Embedder(Embeddings):
    """Hash text tokens into a normalized fixed-size vector."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        return [self.embed(text) for text in texts]

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """LangChain Embeddings interface for document batches."""
        return self.embed_batch(texts)

    def embed_query(self, text: str) -> Vector:
        """LangChain Embeddings interface for single-query retrieval."""
        return self.embed(text)
