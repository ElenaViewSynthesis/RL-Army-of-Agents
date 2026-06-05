"""Helpers for discovering MCP tools."""

import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER = Path(__file__).resolve().parents[1] / "servers" / "fs_server.py"


async def get_mcp_tools() -> list:
    """Discover the initial filesystem tool over stdio."""
    client = MultiServerMCPClient(
        {
            "fs": {
                "command": sys.executable,
                "args": [str(_SERVER)],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()
