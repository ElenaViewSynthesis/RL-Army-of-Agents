"""Trajectory events: the append-only record of everything that happens in a rollout.

Token/loss-mask invariants enforced here (the core training-correctness rule):

- ``token_ids`` are the exact IDs sampled or consumed by the rollout model.
  They are never reconstructed by re-tokenizing decoded text.
- ``loss_mask`` must accompany ``token_ids`` and be the same length, values in {0, 1}.
- Only ``role == ASSISTANT`` events may contain ``loss_mask == 1`` tokens.
  Tool observations, environment responses, templates, system and user messages
  are always fully masked.
- ``rollout_logprobs``, when present, must align 1:1 with ``token_ids``.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventType(StrEnum):
    SYSTEM_MESSAGE = "system_message"
    USER_MESSAGE = "user_message"
    POLICY_OUTPUT = "policy_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ENVIRONMENT_OBSERVATION = "environment_observation"
    VERIFIER_RESULT = "verifier_result"
    BUDGET_VIOLATION = "budget_violation"
    ERROR = "error"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    VERIFIER = "verifier"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TrajectoryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_event_id: str | None = None
    event_type: EventType
    role: Role
    timestamp: datetime = Field(default_factory=_utc_now)
    content: str | None = None
    token_ids: list[int] | None = None
    rollout_logprobs: list[float] | None = None
    loss_mask: list[int] | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: Any = None
    latency_ms: float | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0, description="USD")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_token_alignment(self) -> "TrajectoryEvent":
        if self.token_ids is not None:
            if self.loss_mask is None:
                raise ValueError(
                    "events carrying token_ids must carry an explicit loss_mask; "
                    "masking is never implicit"
                )
            if len(self.loss_mask) != len(self.token_ids):
                raise ValueError(
                    f"loss_mask length {len(self.loss_mask)} != "
                    f"token_ids length {len(self.token_ids)}"
                )
        elif self.loss_mask is not None:
            raise ValueError("loss_mask without token_ids")

        if self.loss_mask is not None:
            bad = {m for m in self.loss_mask} - {0, 1}
            if bad:
                raise ValueError(f"loss_mask values must be 0 or 1, got {sorted(bad)}")
            if self.role is not Role.ASSISTANT and any(self.loss_mask):
                raise ValueError(
                    f"loss_mask=1 tokens on role={self.role.value!r}: only policy-generated "
                    "(assistant) tokens may be selected for optimisation"
                )

        if self.rollout_logprobs is not None:
            if self.token_ids is None:
                raise ValueError("rollout_logprobs without token_ids")
            if len(self.rollout_logprobs) != len(self.token_ids):
                raise ValueError(
                    f"rollout_logprobs length {len(self.rollout_logprobs)} != "
                    f"token_ids length {len(self.token_ids)}"
                )
        return self

    @property
    def num_trainable_tokens(self) -> int:
        return sum(self.loss_mask) if self.loss_mask else 0
