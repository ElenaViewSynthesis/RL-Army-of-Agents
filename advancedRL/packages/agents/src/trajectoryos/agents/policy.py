"""Policy interface for episode loops.

Milestone 3 provides the real SGLang-backed policy that fills ``token_ids``,
``rollout_logprobs`` and ``loss_mask`` with exact sampled values.
``ScriptedPolicy`` is a deterministic stand-in for tests and development: its
tool calls are REALLY executed in the sandbox; only the decision sequence is
pre-scripted. It never fabricates token IDs.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from trajectoryos.schemas import TrajectoryEvent


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class PolicyTurn(BaseModel):
    """One assistant step: optional text, optional tool call, exact token data.

    ``tool_call=None`` means the policy is done and submits its work.
    Token fields are exact sampled values from the rollout engine; they stay
    ``None`` for policies that do not sample tokens (e.g. scripted tests).
    """

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_call: ToolCall | None = None
    token_ids: list[int] | None = None
    rollout_logprobs: list[float] | None = None
    loss_mask: list[int] | None = None
    context_token_ids: list[int] | None = Field(
        default=None,
        description=(
            "Exact token IDs the policy fed to the engine since the previous turn "
            "(prompt template, tool observations). Recorded as a loss_mask=0 "
            "environment event so the trajectory preserves the full token stream."
        ),
    )
    input_tokens_used: int = Field(default=0, ge=0)
    output_tokens_used: int = Field(default=0, ge=0)


class Policy(Protocol):
    def next_turn(self, events: Sequence[TrajectoryEvent]) -> PolicyTurn: ...


class ScriptedPolicy:
    """Deterministic policy for tests/dev: replays a fixed sequence of turns."""

    def __init__(self, turns: Sequence[PolicyTurn]) -> None:
        self._turns = list(turns)
        self._cursor = 0

    def next_turn(self, events: Sequence[TrajectoryEvent]) -> PolicyTurn:
        if self._cursor >= len(self._turns):
            return PolicyTurn(content="(scripted policy exhausted; submitting)")
        turn = self._turns[self._cursor]
        self._cursor += 1
        return turn
