"""Isolated subagent loop with scoped tools and typed return values."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from agent.resilience import apply_limiter, with_retry
from agent.subagent.contract import (
    Finding,
    NamespaceScope,
    SubagentBudgetExceeded,
    SubagentResult,
    SubagentTask,
    ToolScopeViolation,
)

_SYSTEM = SystemMessage(
    content=(
        "You are a focused test-triage subagent. Use only the tools available "
        "to you. Run or inspect tests, identify failures, and return findings. "
        "Do not modify files and do not commit changes."
    )
)

_TEST_TOOLS = frozenset(
    {
        "discover_tests",
        "run_test_file",
        "run_test_node",
        "run_suite",
        "coverage_report",
        "coverage_diff",
        "last_failures",
        "rerun_failed",
    }
)


def _parse_tool_result(raw: Any) -> dict:
    """Normalize LangChain/MCP tool output into a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"output": raw}
        except json.JSONDecodeError:
            return {"output": raw}
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict) and "text" in first:
            return _parse_tool_result(first["text"])
    return {"output": str(raw)}


def _extract_findings(tool_results: list[tuple[str, dict]]) -> list[Finding]:
    """Build findings from test tool outputs."""
    findings: list[Finding] = []
    seen: set[str] = set()

    for tool_name, result in tool_results:
        if tool_name in {"run_suite", "run_test_file", "run_test_node", "rerun_failed"}:
            for item in result.get("results", []):
                outcome = item.get("outcome", "")
                if outcome == "passed":
                    continue
                test_id = item.get("node_id", "unknown")
                if test_id in seen:
                    continue
                seen.add(test_id)
                findings.append(
                    Finding(
                        test_id=test_id,
                        status=outcome if outcome in {"failed", "error", "skipped"} else "error",
                        message=(item.get("output") or result.get("output") or "")[:500],
                    )
                )

        if tool_name == "last_failures":
            for item in result.get("failures", []):
                test_id = item.get("test_id", "unknown")
                if test_id in seen:
                    continue
                seen.add(test_id)
                findings.append(
                    Finding(
                        test_id=test_id,
                        status="failed",
                        message=(item.get("error_text") or "")[:500],
                    )
                )

    return findings


def _extract_artifacts(tool_results: list[tuple[str, dict]]) -> dict[str, str]:
    """Capture compact raw outputs from test tools."""
    artifacts = {}
    for tool_name, result in tool_results:
        if tool_name in _TEST_TOOLS:
            artifacts[f"{tool_name}_output"] = json.dumps(result, sort_keys=True)[:2000]
    return artifacts


class SubagentRunner:
    """Run a scoped, isolated subagent."""

    def __init__(self, all_tools_by_namespace: dict[str, list]) -> None:
        self._all_tools_by_namespace = all_tools_by_namespace

    def _scope_tools(self, scopes: list[NamespaceScope]) -> list:
        """Return only tools allowed by the declared scopes."""
        scoped = []
        for scope in scopes:
            namespace_tools = self._all_tools_by_namespace.get(scope.namespace)
            if namespace_tools is None:
                raise ValueError(f"unknown namespace: {scope.namespace}")
            if scope.tools is None:
                scoped.extend(namespace_tools)
                continue

            by_name = {tool.name: tool for tool in namespace_tools}
            for tool_name in scope.tools:
                if tool_name not in by_name:
                    raise ValueError(f"tool {tool_name!r} not found in namespace {scope.namespace!r}")
                scoped.append(by_name[tool_name])
        return scoped

    async def run(self, task: SubagentTask) -> SubagentResult:
        """Execute the subagent and return a typed result."""
        scoped_tools = self._scope_tools(task.allowed_scopes)
        tool_map = {tool.name: tool for tool in scoped_tools}
        llm = ChatGroq(model=os.environ.get("AGENT_MODEL", "llama-3.1-8b-instant"))
        bound = llm.bind_tools(scoped_tools)

        messages = [_SYSTEM, HumanMessage(content=task.brief)]
        steps = 0
        tokens_used = 0
        tool_results: list[tuple[str, dict]] = []

        while True:
            if steps >= task.budget.max_steps or tokens_used >= task.budget.max_tokens:
                raise SubagentBudgetExceeded(steps, tokens_used, task.budget)

            await apply_limiter()
            response = await with_retry(lambda: bound.ainvoke(messages))
            steps += 1
            tokens_used += (response.usage_metadata or {}).get("total_tokens", 0)
            messages.append(response)

            if not response.tool_calls:
                summary = response.content if isinstance(response.content, str) else ""
                return SubagentResult(
                    status="completed",
                    findings=_extract_findings(tool_results),
                    artifacts=_extract_artifacts(tool_results),
                    tokens_used=tokens_used,
                    steps_taken=steps,
                    summary=summary,
                )

            for call in response.tool_calls:
                tool = tool_map.get(call["name"])
                if tool is None:
                    raise ToolScopeViolation(call["name"], list(tool_map))
                raw = await with_retry(lambda tool=tool, call=call: tool.arun(call["args"]))
                parsed = _parse_tool_result(raw)
                tool_results.append((call["name"], parsed))
                messages.append(ToolMessage(content=str(raw), tool_call_id=call["id"]))
