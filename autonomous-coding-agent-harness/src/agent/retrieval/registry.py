"""Build a retrieval registry from discovered LangChain tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolRegistryEntry:
    namespace: str
    name: str
    description: str
    input_schema: dict[str, Any]


def _namespace_for(tool_name: str) -> str:
    if tool_name.startswith("git_"):
        return "git"
    if tool_name in {
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
        return "ast"
    if tool_name in {
        "discover_tests",
        "run_test_file",
        "run_test_node",
        "run_suite",
        "coverage_report",
        "coverage_diff",
        "last_failures",
        "rerun_failed",
    }:
        return "test"
    if tool_name in {
        "list_dependencies",
        "check_outdated",
        "resolve_import",
        "find_unused_deps",
        "dependency_graph",
        "vulnerability_scan",
        "add_dependency",
    }:
        return "deps"
    if tool_name in {
        "run_linter",
        "run_formatter",
        "run_type_check",
        "build_check",
        "pre_commit_run",
        "run_security_scan",
        "summarize_quality",
    }:
        return "ci"
    return "fs"


def _input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if schema is not None and hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    args = getattr(tool, "args", None)
    return args if isinstance(args, dict) else {}


def build_registry(tools: list[Any]) -> list[ToolRegistryEntry]:
    """Return stable registry entries for discovered tools."""
    entries = []
    for tool in tools:
        name = getattr(tool, "name", "")
        entries.append(
            ToolRegistryEntry(
                namespace=_namespace_for(name),
                name=name,
                description=getattr(tool, "description", "") or "",
                input_schema=_input_schema(tool),
            )
        )
    return entries


def entry_text(entry: ToolRegistryEntry) -> str:
    """Text used for retrieval embedding."""
    return (
        f"namespace: {entry.namespace}\n"
        f"name: {entry.name}\n"
        f"description: {entry.description}\n"
        f"input_schema: {entry.input_schema}"
    )
