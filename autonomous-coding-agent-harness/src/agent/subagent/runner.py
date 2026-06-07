"""Isolated subagent loop with scoped tools and typed return values."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage

from agent.chat_model import get_chat_model_identifier
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


def _create_langchain_agent(model: str, tools: list):
    """Build a LangChain agent for isolated subagent work."""
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        raise RuntimeError(
            "Subagent execution requires langchain. Install it with "
            "`uv pip install langchain` or run `uv sync`."
        ) from exc

    return create_agent(model=model, tools=tools, system_prompt=_SYSTEM.content)


def _content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _usage_tokens(message: Any) -> int:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("total_tokens", 0) or 0)


def _tool_calls(message: Any) -> list[dict]:
    calls = getattr(message, "tool_calls", None)
    return calls if isinstance(calls, list) else []


def _tool_call_id(message: Any) -> str:
    return str(getattr(message, "tool_call_id", "") or "")


def _tool_message_name(message: Any, call_name_by_id: dict[str, str]) -> str:
    name = getattr(message, "name", None)
    if name:
        return str(name)
    return call_name_by_id.get(_tool_call_id(message), "")


def _is_tool_message(message: Any) -> bool:
    return message.__class__.__name__ == "ToolMessage" or bool(_tool_call_id(message))


def _is_ai_message(message: Any) -> bool:
    return message.__class__.__name__ == "AIMessage" or bool(_tool_calls(message))


def _analyze_agent_messages(
    messages: list[Any],
    allowed_tool_names: set[str],
) -> tuple[list[tuple[str, dict]], int, int, str]:
    call_name_by_id: dict[str, str] = {}
    tool_results: list[tuple[str, dict]] = []
    tokens_used = 0
    steps_taken = 0
    summary = ""

    for message in messages:
        tokens_used += _usage_tokens(message)
        if _is_ai_message(message):
            steps_taken += 1
            for call in _tool_calls(message):
                name = str(call.get("name", ""))
                if name and name not in allowed_tool_names:
                    raise ToolScopeViolation(name, sorted(allowed_tool_names))
                call_id = call.get("id")
                if call_id and name:
                    call_name_by_id[str(call_id)] = name

        if _is_tool_message(message):
            name = _tool_message_name(message, call_name_by_id)
            if name:
                tool_results.append((name, _parse_tool_result(_content(message))))

        content = _content(message)
        if isinstance(content, str) and content:
            summary = content

    return tool_results, tokens_used, steps_taken, summary


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
        agent = _create_langchain_agent(
            model=get_chat_model_identifier(),
            tools=scoped_tools,
        )

        await apply_limiter()
        result = await with_retry(
            lambda: agent.ainvoke(
                {"messages": [{"role": "user", "content": task.brief}]},
                config={"recursion_limit": max(task.budget.max_steps * 2 + 1, 2)},
            )
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        tool_results, tokens_used, steps_taken, summary = _analyze_agent_messages(
            messages,
            set(tool_map),
        )
        if steps_taken > task.budget.max_steps or tokens_used > task.budget.max_tokens:
            raise SubagentBudgetExceeded(steps_taken, tokens_used, task.budget)

        return SubagentResult(
            status="completed",
            findings=_extract_findings(tool_results),
            artifacts=_extract_artifacts(tool_results),
            tokens_used=tokens_used,
            steps_taken=steps_taken,
            summary=summary,
        )
