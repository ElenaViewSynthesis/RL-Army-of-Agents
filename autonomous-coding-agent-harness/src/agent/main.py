"""Command-line entry point for the first agent slice."""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph.graph import build_graph
from agent.mcp_client.client import get_mcp_tools_with_namespaces
from agent.retrieval import (
    Embedder,
    InMemoryVectorStore,
    ToolRetriever,
    build_registry,
    entry_text,
)
from agent.subagent import SubagentRunner, make_spawn_subagent_tool

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_REPO = _PROJECT_ROOT / "fixture_repo"

_LONG_HORIZON_TASK = """\
Working in the repository at {repo}:

1. Check git status and list Python files.
2. Read calculator.py and app.py.
3. Find references to the divide function.
4. Add input validation to divide so division by zero and non-numeric inputs raise ValueError.
5. Update app.py to catch ValueError from divide and return None.
6. Run the test suite.
7. Add or update tests for the new validation behavior.
8. If tests fail twice, spawn a test-triage subagent to identify failures.
9. Fix any failures and report the final test result.
"""


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
    tools, tools_by_namespace = await get_mcp_tools_with_namespaces()
    runner = SubagentRunner(tools_by_namespace)
    tools = [*tools, make_spawn_subagent_tool(runner)]
    entries = build_registry(tools)
    embedder = Embedder()
    store = InMemoryVectorStore()
    store.upsert(entries, embedder.embed_batch([entry_text(entry) for entry in entries]))
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

    _print_trace(result["messages"])
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "(no answer)"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "long_horizon":
        if not _FIXTURE_REPO.exists():
            raise FileNotFoundError(f"fixture repo not found: {_FIXTURE_REPO}")
        task = _LONG_HORIZON_TASK.format(repo=_FIXTURE_REPO)
    else:
        spec_path = _PROJECT_ROOT / "SPEC.md"
        task = f"Read the file {spec_path} and tell me its title."
    answer = asyncio.run(run(task))
    print(answer)


if __name__ == "__main__":
    main()
