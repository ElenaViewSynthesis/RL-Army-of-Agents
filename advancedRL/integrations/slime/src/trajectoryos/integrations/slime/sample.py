"""Trajectory → slime ``Sample`` conversion.

Correctness rules enforced here (tested in ``tests/trajectory_correctness``):

- ``tokens`` is the exact flattened token stream from the trajectory —
  never re-tokenized text.
- The prompt region is everything before the first policy (assistant) tokens;
  the response region is everything from there on, so multi-turn observations
  land inside the response with ``loss_mask=0``.
- ``loss_mask`` and ``rollout_log_probs`` cover the response region and align
  1:1 with its tokens. Every trainable token (mask=1) MUST have a real rollout
  logprob; missing ones are a hard error, not a silent fill. Non-trainable
  response tokens (observations) get ``masked_logprob_fill`` — they never
  contribute to the loss.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trajectoryos.schemas import Role, TerminalState, Trajectory


class SampleConversionError(ValueError):
    """A trajectory violates the token/mask/logprob invariants for training."""


_STATUS_BY_TERMINAL_STATE: dict[TerminalState, str] = {
    TerminalState.COMPLETED: "COMPLETED",
    TerminalState.TRUNCATED: "TRUNCATED",
    TerminalState.BUDGET_EXCEEDED: "TRUNCATED",
    TerminalState.ERROR: "ABORTED",
    TerminalState.ABORTED: "ABORTED",
}


class SlimeSampleData(BaseModel):
    """Dependency-free mirror of the ``slime.utils.types.Sample`` fields we fill.

    Converted to a real slime ``Sample`` by :func:`to_slime_sample` in an
    environment where slime is installed.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    tokens: list[int]
    response_length: int = Field(ge=0)
    response: str
    reward: float | None = None
    loss_mask: list[int]
    rollout_log_probs: list[float]
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _aligned(self) -> "SlimeSampleData":
        if self.response_length > len(self.tokens):
            raise ValueError("response_length exceeds token count")
        if len(self.loss_mask) != self.response_length:
            raise ValueError(
                f"loss_mask length {len(self.loss_mask)} != response_length {self.response_length}"
            )
        if len(self.rollout_log_probs) != self.response_length:
            raise ValueError(
                f"rollout_log_probs length {len(self.rollout_log_probs)} != "
                f"response_length {self.response_length}"
            )
        return self


def trajectory_to_sample_data(
    trajectory: Trajectory,
    *,
    masked_logprob_fill: float = 0.0,
) -> SlimeSampleData:
    tokens, mask, logprobs = trajectory.token_stream()
    if not tokens:
        raise SampleConversionError(
            f"trajectory {trajectory.rollout_id} carries no tokens; "
            "it cannot become a training sample"
        )

    prompt_length = _prompt_length(trajectory)
    response_mask = mask[prompt_length:]
    response_logprobs: list[float] = []
    for offset, (m, lp) in enumerate(zip(response_mask, logprobs[prompt_length:], strict=True)):
        if m == 1:
            if lp is None:
                raise SampleConversionError(
                    f"trainable token at position {prompt_length + offset} has no rollout "
                    "logprob; exact sampled logprobs are required for training"
                )
            response_logprobs.append(lp)
        else:
            response_logprobs.append(masked_logprob_fill)

    prompt_text, response_text = _texts(trajectory)
    return SlimeSampleData(
        prompt=prompt_text,
        tokens=tokens,
        response_length=len(tokens) - prompt_length,
        response=response_text,
        loss_mask=response_mask,
        rollout_log_probs=response_logprobs,
        status=_STATUS_BY_TERMINAL_STATE[trajectory.terminal_state],
        metadata={
            "run_id": trajectory.run_id,
            "rollout_id": trajectory.rollout_id,
            "task_id": trajectory.task_id,
            "policy_version": trajectory.policy_version,
            "terminal_state": trajectory.terminal_state.value,
            "termination_reason": trajectory.termination_reason,
            "reward_components": trajectory.reward_components,
            "cost_summary": trajectory.cost_summary.model_dump(),
        },
    )


def to_slime_sample(data: SlimeSampleData) -> Any:
    """Build a real ``slime.utils.types.Sample`` (requires slime installed)."""
    try:
        from slime.utils.types import Sample
    except ImportError as exc:  # pragma: no cover - exercised only with slime installed
        raise ImportError(
            "slime is not installed in this environment. Install THUDM/slime "
            "(GPU training environment) to convert SlimeSampleData into Sample objects."
        ) from exc
    sample = Sample(
        prompt=data.prompt,
        tokens=data.tokens,
        response_length=data.response_length,
        response=data.response,
        reward=data.reward,
        loss_mask=data.loss_mask,
        rollout_log_probs=data.rollout_log_probs,
        metadata=data.metadata,
    )
    sample.status = getattr(Sample.Status, data.status)
    return sample


def _prompt_length(trajectory: Trajectory) -> int:
    """Token count before the first assistant tokens (the prompt region)."""
    length = 0
    for event in trajectory.events:
        if event.token_ids is None:
            continue
        if event.role is Role.ASSISTANT:
            return length
        length += len(event.token_ids)
    raise SampleConversionError(
        f"trajectory {trajectory.rollout_id} has no assistant tokens; there is nothing to train on"
    )


def _texts(trajectory: Trajectory) -> tuple[str, str]:
    prompt_parts: list[str] = []
    response_parts: list[str] = []
    seen_assistant_tokens = False
    for event in trajectory.events:
        if event.role is Role.ASSISTANT and event.token_ids is not None:
            seen_assistant_tokens = True
        if event.content:
            (response_parts if seen_assistant_tokens else prompt_parts).append(event.content)
    return "\n".join(prompt_parts), "\n".join(response_parts)
