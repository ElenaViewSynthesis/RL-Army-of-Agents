"""Command-line entry point for the first agent slice."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from agent.graph.graph import build_graph
from agent.mcp_client.client import get_mcp_tools

load_dotenv()


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

    for message in reversed(result["messages"]):
        if getattr(message, "content", None):
            return str(message.content)
    return "(no answer)"


def main() -> None:
    spec_path = Path(__file__).resolve().parents[2] / "SPEC.md"
    task = f"Read the file {spec_path} and tell me its title."
    answer = asyncio.run(run(task))
    print(answer)


if __name__ == "__main__":
    main()
