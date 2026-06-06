"""Deterministic local text embedder for tool retrieval.

The production target is sentence-transformer embeddings stored in pgvector.
This milestone starts with a small hashing embedder so retrieval behavior is
testable without downloading a model or running a database.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:  # type: ignore[no-redef]
        """Fallback base for local retrieval tests without LangChain installed."""


Vector = list[float]

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class EmbeddingService(Protocol):
    """Embedding contract shared by local and LangChain-backed retrieval."""

    dimensions: int

    def embed(self, text: str) -> Vector: ...

    def embed_batch(self, texts: list[str]) -> list[Vector]: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


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


try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]


class TransformerEmbedder(Embeddings):
    """Real sentence-transformers embedder for live retrieval.

    This class uses `sentence-transformers` to produce dense vectors. It is
    intended for use in production or live evaluation where the real semantic
    embeddings are required. The lightweight `Embedder` (hash-based) remains
    available for fast unit tests.
    """

    def __init__(self, model_name: str | None = None) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; install sentence-transformers"
            )
        self._model_name = model_name or os.environ.get(
            "SENTENCE_TRANSFORMER_MODEL",
            "all-MiniLM-L6-v2",
        )
        self._model = SentenceTransformer(self._model_name)
        try:
            self.dimensions = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback: infer from a sample encoding
            sample = self._model.encode("_", convert_to_numpy=True)
            self.dimensions = int(len(sample))

    def _normalize(self, vec: list[float] | "numpy.ndarray") -> Vector:
        v = list(vec)
        norm = math.sqrt(sum(float(x) * float(x) for x in v))
        if norm == 0:
            return [float(x) for x in v]
        return [float(x) / norm for x in v]

    def embed(self, text: str) -> Vector:
        arr = self._model.encode(text, convert_to_numpy=True)
        return self._normalize(arr)

    def embed_batch(self, texts: list[str]) -> list[Vector]:
        arrs = self._model.encode(texts, convert_to_numpy=True)
        return [self._normalize(arr) for arr in arrs]

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return self.embed_batch(texts)

    def embed_query(self, text: str) -> Vector:
        return self.embed(text)
