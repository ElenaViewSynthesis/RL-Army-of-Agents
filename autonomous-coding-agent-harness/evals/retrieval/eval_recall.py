"""Recall@k evaluation for the local tool retriever."""

from __future__ import annotations

from dataclasses import dataclass

from agent.retrieval import Embedder, InMemoryVectorStore, ToolRegistryEntry, ToolRetriever, entry_text
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
    store.upsert(entries, embedder.embed_batch([entry_text(entry) for entry in entries]))
    retriever = ToolRetriever(store, embedder)

    for k in (5, 8, 12):
        hits = 0
        for goal, expected in PAIRS:
            if expected in retriever.retrieve(goal, k=k):
                hits += 1
        print(f"recall@{k} = {hits / len(PAIRS):.3f} ({hits}/{len(PAIRS)})")


if __name__ == "__main__":
    main()
