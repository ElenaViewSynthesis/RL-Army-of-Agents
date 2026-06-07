from agent.retrieval import Embedder, InMemoryVectorStore, ToolRetriever
from agent.retrieval.registry import ToolRegistryEntry, entry_text


def _retriever() -> ToolRetriever:
    entries = [
        ToolRegistryEntry("fs", "read_file", "read file contents", {}),
        ToolRegistryEntry("fs", "grep", "search text inside files", {}),
        ToolRegistryEntry("git", "git_status", "show git status", {}),
        ToolRegistryEntry("git", "git_commit", "commit staged changes", {}),
    ]
    embedder = Embedder()
    store = InMemoryVectorStore()
    store.upsert(entries, embedder.embed_batch([entry_text(entry) for entry in entries]))
    return ToolRetriever(store, embedder)


def test_retrieve_returns_relevant_tool() -> None:
    retriever = _retriever()

    names = retriever.retrieve("search text in files", k=1)

    assert "grep" in names


def test_retrieve_always_includes_core_tools() -> None:
    retriever = _retriever()

    names = retriever.retrieve("commit the changes", k=1)

    assert "read_file" in names
    assert "git_status" in names


def test_retrieve_wider_increases_k() -> None:
    retriever = _retriever()

    _, new_k = retriever.retrieve_wider("commit the changes", current_k=1)

    assert new_k > 1
