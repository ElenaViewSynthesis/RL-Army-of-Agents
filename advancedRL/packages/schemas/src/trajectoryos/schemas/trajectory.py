"""Completed (or failed) rollout: an ordered list of events plus outcome and costs.

Failed and truncated trajectories are first-class data: ``terminal_state`` and
``termination_reason`` record why a rollout ended; nothing is silently discarded.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from trajectoryos.schemas.events import TrajectoryEvent


class TerminalState(StrEnum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"
    ABORTED = "aborted"


class CostSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: dict[str, int] = Field(default_factory=dict)
    wallclock_seconds: float = Field(default=0.0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    logs: str | None = None


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    events: list[TrajectoryEvent] = Field(default_factory=list)
    terminal_state: TerminalState
    termination_reason: str | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)
    total_reward: float = 0.0
    verifier_result: VerifierResult | None = None
    cost_summary: CostSummary = Field(default_factory=CostSummary)

    def token_stream(self) -> tuple[list[int], list[int], list[float | None]]:
        """Flatten events (in order) into aligned token/mask/logprob sequences.

        Returns ``(token_ids, loss_mask, rollout_logprobs)`` where all three lists
        have equal length; logprob entries are ``None`` for tokens the rollout
        engine did not score (e.g. re-ingested observation tokens).
        This is the canonical input for building slime ``Sample`` objects —
        training targets are taken from here, never from re-tokenized text.
        """
        token_ids: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float | None] = []
        for event in self.events:
            if event.token_ids is None:
                continue
            token_ids.extend(event.token_ids)
            # Event validators guarantee mask presence and alignment.
            assert event.loss_mask is not None
            loss_mask.extend(event.loss_mask)
            if event.rollout_logprobs is not None:
                logprobs.extend(event.rollout_logprobs)
            else:
                logprobs.extend([None] * len(event.token_ids))
        return token_ids, loss_mask, logprobs

    @property
    def num_trainable_tokens(self) -> int:
        return sum(e.num_trainable_tokens for e in self.events)
