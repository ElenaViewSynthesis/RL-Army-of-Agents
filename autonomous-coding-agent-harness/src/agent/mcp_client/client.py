"""Helpers for discovering MCP tools."""

import copy
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
_FS_SERVER = _SERVERS_DIR / "fs_server.py"
_GIT_SERVER = _SERVERS_DIR / "git_server.py"
_AST_SERVER = _SERVERS_DIR / "ast_server.py"
_TEST_SERVER = _SERVERS_DIR / "test_server.py"
_DEPS_SERVER = _SERVERS_DIR / "deps_server.py"
_CI_SERVER = _SERVERS_DIR / "ci_server.py"

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
    "ast": {
        "command": sys.executable,
        "args": [str(_AST_SERVER)],
        "transport": "stdio",
    },
    "test": {
        "command": sys.executable,
        "args": [str(_TEST_SERVER)],
        "transport": "stdio",
    },
    "deps": {
        "command": sys.executable,
        "args": [str(_DEPS_SERVER)],
        "transport": "stdio",
    },
    "ci": {
        "command": sys.executable,
        "args": [str(_CI_SERVER)],
        "transport": "stdio",
    },
}


async def get_mcp_tools() -> list:
    """Discover all MCP tool namespaces over stdio."""
    client = MultiServerMCPClient(copy.deepcopy(_CONNECTIONS))
    return await client.get_tools()
