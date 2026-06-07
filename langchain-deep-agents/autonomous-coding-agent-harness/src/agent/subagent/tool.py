"""Parent-facing spawn_subagent tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from agent.subagent.contract import NamespaceScope, SubagentBudget, SubagentResult, SubagentTask
from agent.subagent.runner import SubagentRunner

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool


class SpawnSubagentInput(BaseModel):
    """Input schema for the parent-facing subagent tool."""

    brief: str = Field(min_length=1)
    allowed_scopes: list[NamespaceScope] = Field(
        default_factory=lambda: [
            NamespaceScope(namespace="test"),
            NamespaceScope(namespace="fs", tools=["read_file"]),
        ]
    )
    max_steps: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=20_000, ge=1)


def make_spawn_subagent_tool(runner: SubagentRunner) -> StructuredTool:
    """Create the `spawn_subagent` LangChain tool."""
    try:
        from langchain.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            "Subagent tool wrapping requires langchain. Install it with "
            "`uv pip install langchain` or run `uv sync`."
        ) from exc

    @tool(
        "spawn_subagent",
        description=(
            "Launch an isolated test-triage subagent with fresh context, "
            "scoped tools, its own budget, and a structured result."
        ),
        args_schema=SpawnSubagentInput,
    )
    async def spawn_subagent(
        brief: str,
        allowed_scopes: list[NamespaceScope] | None = None,
        max_steps: int = 8,
        max_tokens: int = 20_000,
    ) -> dict:
        scopes = allowed_scopes or [
            NamespaceScope(namespace="test"),
            NamespaceScope(namespace="fs", tools=["read_file"]),
        ]
        task = SubagentTask(
            brief=brief,
            allowed_scopes=scopes,
            budget=SubagentBudget(max_steps=max_steps, max_tokens=max_tokens),
        )
        try:
            result = await runner.run(task)
        except Exception as exc:
            result = SubagentResult(
                status="error",
                findings=[],
                artifacts={},
                tokens_used=0,
                steps_taken=0,
                error=str(exc),
            )
        return result.model_dump()

    return spawn_subagent
