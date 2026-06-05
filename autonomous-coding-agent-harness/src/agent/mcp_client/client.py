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


async def get_mcp_tools_with_namespaces() -> tuple[list, dict[str, list]]:
    """Discover all tools and return them grouped by namespace."""
    client = MultiServerMCPClient(copy.deepcopy(_CONNECTIONS))
    tools = await client.get_tools()
    by_namespace = {namespace: [] for namespace in _CONNECTIONS}
    for tool in tools:
        name = getattr(tool, "name", "")
        if name.startswith("git_"):
            by_namespace["git"].append(tool)
        elif name in {
            "parse_module",
            "list_symbols",
            "find_definition",
            "find_references",
            "list_imports",
            "compute_complexity",
            "detect_dead_code",
            "extract_function_signature",
            "find_unused_imports",
        }:
            by_namespace["ast"].append(tool)
        elif name in {
            "discover_tests",
            "run_test_file",
            "run_test_node",
            "run_suite",
            "coverage_report",
            "coverage_diff",
            "last_failures",
            "rerun_failed",
        }:
            by_namespace["test"].append(tool)
        elif name in {
            "list_dependencies",
            "check_outdated",
            "resolve_import",
            "find_unused_deps",
            "dependency_graph",
            "vulnerability_scan",
            "add_dependency",
        }:
            by_namespace["deps"].append(tool)
        elif name in {
            "run_linter",
            "run_formatter",
            "run_type_check",
            "build_check",
            "pre_commit_run",
            "run_security_scan",
            "summarize_quality",
        }:
            by_namespace["ci"].append(tool)
        else:
            by_namespace["fs"].append(tool)
    return tools, by_namespace
