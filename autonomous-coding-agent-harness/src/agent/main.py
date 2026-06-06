"""Command-line entry point for the first agent slice."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.errors import RateLimitExceeded, RequestTooLarge, RetryExhausted
from agent.graph.graph import build_graph
from agent.logging_config import configure_logging, get_logger
from agent.mcp_client.client import get_mcp_tools_with_namespaces
from agent.retrieval import (
    Embedder,
    TransformerEmbedder,
    InMemoryVectorStore,
    PgVectorStore,
    ToolRetriever,
    build_registry,
    entry_text,
)
from agent.subagent import SubagentRunner, make_spawn_subagent_tool

load_dotenv()
configure_logging()
_log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_store(entries, embedder: Embedder):
    """Build the configured retrieval store."""
    texts = [entry_text(entry) for entry in entries]
    embeddings = embedder.embed_batch(texts)
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        store = PgVectorStore(database_url, embedder)
        store.init_schema()
        store.upsert(entries, embeddings)
        _log.info("using pgvector retrieval store")
        return store

    store = InMemoryVectorStore()
    store.upsert(entries, embeddings)
    _log.info("using in-memory retrieval store")
    return store


def _print_trace(messages: list) -> None:
    """Print a compact trace that shows whether the agent used tools."""
    print("\nTRACE")
    print("=" * 60)
    for index, message in enumerate(messages):
        tag = f"[{index}]"
        if isinstance(message, SystemMessage):
            print(f"{tag} system: {message.content[:120]}")
        elif isinstance(message, HumanMessage):
            print(f"{tag} human: {message.content}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            print(f"{tag} ai tool call:")
            for call in message.tool_calls:
                print(f"  - {call['name']} {json.dumps(call['args'], sort_keys=True)}")
        elif isinstance(message, AIMessage):
            print(f"{tag} ai final: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"{tag} tool result: {str(message.content)[:240]}")
        else:
            print(f"{tag} {type(message).__name__}: {str(message)[:120]}")
    print("=" * 60)


async def run(task: str) -> str:
    """Run the minimal agent against a plain-English task."""
    try:
        tools, tools_by_namespace = await get_mcp_tools_with_namespaces()
        runner = SubagentRunner(tools_by_namespace)
        tools = [*tools, make_spawn_subagent_tool(runner)]
        entries = build_registry(tools)
        # Prefer the real sentence-transformers embedder when a real
        # pgvector-backed database is configured for live retrieval. Keep the
        # hash-based `Embedder` for fast unit tests and local runs.
        database_url = os.environ.get("DATABASE_URL")
        if database_url:
            try:
                embedder = TransformerEmbedder()
            except Exception:
                embedder = Embedder()
        else:
            embedder = Embedder()
        store = _build_store(entries, embedder)
        retriever = ToolRetriever(store, embedder)
        graph = build_graph(tools, retriever)
        result = await graph.ainvoke(
            {
                "task": task,
                "plan": "",
                "tools": tools,
                "available_tool_names": [],
                "retrieval_k": 8,
                "retrieval_miss_count": 0,
                "progress_ledger": "",
                "token_estimate": 0,
                "compaction_count": 0,
                "ledger_message_id": None,
                "messages": [],
            }
        )
    except RateLimitExceeded as exc:
        _log.error("rate limit exceeded", extra={"args_preview": str(exc)})
        return "(rate limit exceeded)"
    except RequestTooLarge as exc:
        _log.error("request too large", extra={"args_preview": str(exc)})
        return "(request too large)"
    except RetryExhausted as exc:
        _log.error("retry exhausted", extra={"args_preview": str(exc)})
        return f"(retry exhausted: {exc.last_error})"

    _print_trace(result["messages"])
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "(no answer)"


def main() -> None:
    spec_path = _PROJECT_ROOT / "SPEC.md"
    task = f"Read the file {spec_path} and tell me its title."
    answer = asyncio.run(run(task))
    print(answer)


if __name__ == "__main__":
    main()
