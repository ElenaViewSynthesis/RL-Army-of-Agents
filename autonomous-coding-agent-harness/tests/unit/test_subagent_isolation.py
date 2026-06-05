from pydantic import ValidationError

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
