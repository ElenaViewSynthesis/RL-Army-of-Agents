"""Command-line entry point for the first agent slice."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph.graph import build_graph
from agent.mcp_client.client import get_mcp_tools

load_dotenv()


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
    tools = await get_mcp_tools()
    graph = build_graph(tools)
    result = await graph.ainvoke(
        {
            "task": task,
            "plan": "",
            "tools": tools,
            "available_tool_names": [],
            "messages": [],
        }
    )

    _print_trace(result["messages"])
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "(no answer)"


def main() -> None:
    spec_path = Path(__file__).resolve().parents[2] / "SPEC.md"
    task = f"Read the file {spec_path} and tell me its title."
    answer = asyncio.run(run(task))
    print(answer)


if __name__ == "__main__":
    main()
