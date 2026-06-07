import pytest

pytest.importorskip("langchain_core")

from agent.subagent import SubagentRunner, make_spawn_subagent_tool


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.integration
def test_spawn_subagent_tool_schema_is_registered() -> None:
    runner = SubagentRunner({"test": [FakeTool("run_suite")], "fs": [FakeTool("read_file")]})

    tool = make_spawn_subagent_tool(runner)

    assert tool.name == "spawn_subagent"
    assert "subagent" in tool.description.lower()
