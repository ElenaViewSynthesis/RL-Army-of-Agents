"""Recall@k evaluation for the local tool retriever."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

from agent.retrieval import (
    Embedder,
    InMemoryVectorStore,
    PgVectorStore,
    ToolRegistryEntry,
    ToolRetriever,
    TransformerEmbedder,
    entry_text,
)
from evals.retrieval.labeled_pairs import PAIRS


@dataclass
class EvalTool:
    name: str
    description: str
    args: dict


def _tools_from_pairs() -> list[EvalTool]:
    names = sorted({tool for _, tool in PAIRS})
    return [
        EvalTool(
            name=name,
            description=name.replace("_", " "),
            args={"query": {"type": "string"}},
        )
        for name in names
    ]


def _registry(tools: list[EvalTool]) -> list[ToolRegistryEntry]:
    return [
        ToolRegistryEntry(
            namespace="git" if tool.name.startswith("git_") else "fs",
            name=tool.name,
            description=tool.description,
            input_schema=tool.args,
        )
        for tool in tools
    ]


def main() -> None:
    entries = _registry(_tools_from_pairs())
    embedder = Embedder()
    store = InMemoryVectorStore()
    texts = [entry_text(entry) for entry in entries]
    store.upsert(entries, embedder.embed_batch(texts))
    retriever = ToolRetriever(store, embedder)

    print("InMemoryVectorStore results:")
    for k in (5, 8, 12):
        hits = 0
        for goal, expected in PAIRS:
            if expected in retriever.retrieve(goal, k=k):
                hits += 1
        print(f"recall@{k} = {hits / len(PAIRS):.3f} ({hits}/{len(PAIRS)})")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return

    print("\nDATABASE_URL detected - running pgvector evaluation...")
    try:
        real_embedder = TransformerEmbedder()
        model_name = getattr(real_embedder, "_model_name", "sentence-transformers")
    except Exception as exc:
        print(
            "Failed to initialize TransformerEmbedder: "
            f"{exc}; falling back to hash Embedder."
        )
        real_embedder = Embedder()
        model_name = "hash-embedder"

    pg_store = PgVectorStore(database_url, real_embedder)
    pg_store.init_schema()
    pg_store.upsert(entries, real_embedder.embed_batch(texts))
    pg_retriever = ToolRetriever(pg_store, real_embedder)

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": "tool_registry_pairs",
        "model": model_name,
        "store": "pgvector",
        "metrics": {},
    }

    for k in (5, 8, 12):
        hits = 0
        for goal, expected in PAIRS:
            if expected in pg_retriever.retrieve(goal, k=k):
                hits += 1
        value = hits / len(PAIRS)
        results["metrics"][str(k)] = {
            "recall": value,
            "hits": hits,
            "total": len(PAIRS),
        }
        print(f"pgvector recall@{k} = {value:.3f} ({hits}/{len(PAIRS)})")

    out_path = os.path.join(
        os.path.dirname(__file__),
        "pgvector_recall_results.jsonl",
    )
    try:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(results) + "\n")
        print(f"Recorded pgvector recall results to {out_path}")
    except Exception as exc:
        print(f"Failed to write pgvector results: {exc}")


if __name__ == "__main__":
    main()
