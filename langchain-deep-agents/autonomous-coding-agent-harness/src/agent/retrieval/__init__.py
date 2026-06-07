"""Tool retrieval primitives."""

from agent.retrieval.embedder import Embedder, EmbeddingService, TransformerEmbedder
from agent.retrieval.registry import ToolRegistryEntry, build_registry, entry_text
from agent.retrieval.retriever import ToolRetriever
from agent.retrieval.store import InMemoryVectorStore, PgVectorStore

__all__ = [
    "Embedder",
    "EmbeddingService",
    "TransformerEmbedder",
    "InMemoryVectorStore",
    "PgVectorStore",
    "ToolRegistryEntry",
    "ToolRetriever",
    "build_registry",
    "entry_text",
]
