import asyncio

from langchain_core.messages import AIMessage
from pydantic import ValidationError

import agent.subagent.runner as runner_module
from agent.subagent.contract import NamespaceScope, SubagentBudget, SubagentTask
from agent.subagent.runner import SubagentRunner


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


def test_subagent_task_requires_brief() -> None:
    try:
        SubagentTask(brief="", allowed_scopes=[NamespaceScope(namespace="test")])
    except ValidationError:
        return

    raise AssertionError("empty subagent brief should fail validation")


def test_subagent_budget_requires_positive_steps() -> None:
    try:
        SubagentBudget(max_steps=0)
    except ValidationError:
        return

    raise AssertionError("zero max_steps should fail validation")


def test_scope_tools_allows_entire_namespace() -> None:
    runner = SubagentRunner({"test": [FakeTool("run_suite"), FakeTool("last_failures")]})

    scoped = runner._scope_tools([NamespaceScope(namespace="test")])

    assert [tool.name for tool in scoped] == ["run_suite", "last_failures"]


def test_scope_tools_allows_named_subset() -> None:
    runner = SubagentRunner(
        {
            "test": [FakeTool("run_suite")],
            "fs": [FakeTool("read_file"), FakeTool("write_file")],
        }
    )

    scoped = runner._scope_tools(
        [
            NamespaceScope(namespace="test"),
            NamespaceScope(namespace="fs", tools=["read_file"]),
        ]
    )

    assert [tool.name for tool in scoped] == ["run_suite", "read_file"]


def test_scope_tools_rejects_unknown_namespace() -> None:
    runner = SubagentRunner({"test": [FakeTool("run_suite")]})

    try:
        runner._scope_tools([NamespaceScope(namespace="git")])
    except ValueError as exc:
        assert "unknown namespace" in str(exc)
        return

    raise AssertionError("unknown namespace should fail")


def test_scope_tools_rejects_unknown_tool() -> None:
    runner = SubagentRunner({"fs": [FakeTool("read_file")]})

    try:
        runner._scope_tools([NamespaceScope(namespace="fs", tools=["write_file"])])
    except ValueError as exc:
        assert "not found" in str(exc)
        return

    raise AssertionError("unknown scoped tool should fail")


def test_run_uses_create_agent_with_scoped_tools(monkeypatch) -> None:
    created = {}

    class FakeAgent:
        async def ainvoke(self, payload, config=None):
            created["payload"] = payload
            created["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="triage complete",
                        usage_metadata={
                            "input_tokens": 5,
                            "output_tokens": 12,
                            "total_tokens": 17,
                        },
                    )
                ]
            }

    def fake_create_agent(model, tools):
        created["model"] = model
        created["tools"] = tools
        return FakeAgent()

    monkeypatch.setattr(
        runner_module,
        "get_chat_model_identifier",
        lambda: "google_genai:gemini-3.5-flash",
    )
    monkeypatch.setattr(runner_module, "_create_langchain_agent", fake_create_agent)

    runner = SubagentRunner(
        {
            "test": [FakeTool("run_suite")],
            "fs": [FakeTool("read_file"), FakeTool("write_file")],
        }
    )
    task = SubagentTask(
        brief="triage tests",
        allowed_scopes=[
            NamespaceScope(namespace="test"),
            NamespaceScope(namespace="fs", tools=["read_file"]),
        ],
    )

    result = asyncio.run(runner.run(task))

    assert created["model"] == "google_genai:gemini-3.5-flash"
    assert [tool.name for tool in created["tools"]] == ["run_suite", "read_file"]
    assert created["payload"] == {
        "messages": [{"role": "user", "content": "triage tests"}]
    }
    assert created["config"]["recursion_limit"] == 17
    assert result.status == "completed"
    assert result.summary == "triage complete"
    assert result.tokens_used == 17
    assert result.steps_taken == 1
