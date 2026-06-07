"""Pydantic contracts and typed errors for subagents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NamespaceScope(BaseModel):
    """Tools a subagent may access from one namespace."""

    namespace: str = Field(min_length=1)
    tools: list[str] | None = None


class SubagentBudget(BaseModel):
    """Separate execution budget for a subagent."""

    max_steps: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=20_000, ge=1)


class SubagentTask(BaseModel):
    """Task given to an isolated subagent."""

    brief: str = Field(min_length=1)
    allowed_scopes: list[NamespaceScope]
    budget: SubagentBudget = Field(default_factory=SubagentBudget)


class Finding(BaseModel):
    """Structured test-triage finding."""

    test_id: str
    status: Literal["failed", "error", "skipped"]
    message: str
    file_path: str | None = None
    line: int | None = Field(default=None, ge=1)


class SubagentResult(BaseModel):
    """Typed result consumed by the parent agent."""

    status: Literal["completed", "budget_exceeded", "error"]
    findings: list[Finding]
    artifacts: dict[str, str]
    tokens_used: int = Field(ge=0)
    steps_taken: int = Field(ge=0)
    summary: str = ""
    error: str | None = None


class SubagentBudgetExceeded(RuntimeError):
    """Raised when a subagent exceeds its budget."""

    def __init__(self, steps_taken: int, tokens_used: int, budget: SubagentBudget) -> None:
        self.steps_taken = steps_taken
        self.tokens_used = tokens_used
        self.budget = budget
        super().__init__(
            "subagent budget exceeded: "
            f"{steps_taken}/{budget.max_steps} steps, "
            f"{tokens_used}/{budget.max_tokens} tokens"
        )


class ToolScopeViolation(RuntimeError):
    """Raised when a subagent attempts to call a non-scoped tool."""

    def __init__(self, tool_name: str, allowed: list[str]) -> None:
        self.tool_name = tool_name
        self.allowed = allowed
        super().__init__(
            f"tool {tool_name!r} is outside the subagent scope "
            f"(allowed: {sorted(allowed)})"
        )
