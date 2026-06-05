"""Helpers for discovering MCP tools."""

import copy
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
_FS_SERVER = _SERVERS_DIR / "fs_server.py"
_GIT_SERVER = _SERVERS_DIR / "git_server.py"

_CONNECTIONS = {
    "fs": {
        "command": sys.executable,
        "args": [str(_FS_SERVER)],
        "transport": "stdio",
    },
    "git": {
        "command": sys.executable,
        "args": [str(_GIT_SERVER)],
        "transport": "stdio",
    },
}


async def get_mcp_tools() -> list:
    """Discover filesystem and git tools over stdio."""
    client = MultiServerMCPClient(copy.deepcopy(_CONNECTIONS))
    return await client.get_tools()
