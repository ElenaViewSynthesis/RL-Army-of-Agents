"""Token-ID and loss-mask correctness: the core training-data invariant.

Trainable trajectories preserve exact sampled token IDs; only policy-generated
(assistant) tokens may carry loss_mask=1; observations/templates/user text are
always masked out.
"""

import pytest
from pydantic import ValidationError
from trajectoryos.schemas import (
    EventType,
    Role,
    TerminalState,
    Trajectory,
    TrajectoryEvent,
)


def policy_event(tokens: list[int], mask: list[int], logprobs: list[float]) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_type=EventType.POLICY_OUTPUT,
        role=Role.ASSISTANT,
        token_ids=tokens,
        loss_mask=mask,
        rollout_logprobs=logprobs,
    )


def observation_event(role: Role, event_type: EventType, tokens: list[int]) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_type=event_type,
        role=role,
        token_ids=tokens,
        loss_mask=[0] * len(tokens),
    )


class TestEventLevelInvariants:
    @pytest.mark.parametrize(
        ("role", "event_type"),
        [
            (Role.TOOL, EventType.TOOL_RESULT),
            (Role.ENVIRONMENT, EventType.ENVIRONMENT_OBSERVATION),
            (Role.USER, EventType.USER_MESSAGE),
            (Role.SYSTEM, EventType.SYSTEM_MESSAGE),
            (Role.VERIFIER, EventType.VERIFIER_RESULT),
        ],
    )
    def test_non_policy_tokens_must_be_fully_masked(
        self, role: Role, event_type: EventType
    ) -> None:
        # loss_mask=0 everywhere: fine.
        observation_event(role, event_type, [5, 6, 7])
        # any loss_mask=1: rejected at construction.
        with pytest.raises(ValidationError, match="only policy-generated"):
            TrajectoryEvent(
                event_type=event_type,
                role=role,
                token_ids=[5, 6, 7],
                loss_mask=[0, 1, 0],
            )

    def test_assistant_template_tokens_may_be_masked(self) -> None:
        """Chat-template scaffolding inside an assistant turn is maskable to 0."""
        event = policy_event([100, 7, 8, 9], [0, 1, 1, 1], [-0.0, -0.3, -0.2, -0.9])
        assert event.num_trainable_tokens == 3


class TestTrajectoryFlattening:
    def make_multi_turn_trajectory(self) -> Trajectory:
        return Trajectory(
            run_id="run-1",
            rollout_id="ro-1",
            task_id="t-1",
            policy_version="v0",
            terminal_state=TerminalState.COMPLETED,
            events=[
                observation_event(Role.SYSTEM, EventType.SYSTEM_MESSAGE, [1, 2]),
                observation_event(Role.USER, EventType.USER_MESSAGE, [3, 4]),
                policy_event([10, 11], [1, 1], [-0.5, -0.1]),
                TrajectoryEvent(  # tool call bookkeeping event without tokens
                    event_type=EventType.TOOL_CALL,
                    role=Role.ASSISTANT,
                    tool_name="run_shell",
                    tool_arguments={"cmd": "pytest -q"},
                ),
                observation_event(Role.TOOL, EventType.TOOL_RESULT, [20, 21, 22]),
                policy_event([30], [1], [-0.9]),
            ],
        )

    def test_flatten_preserves_order_and_alignment(self) -> None:
        tokens, mask, logprobs = self.make_multi_turn_trajectory().token_stream()
        assert tokens == [1, 2, 3, 4, 10, 11, 20, 21, 22, 30]
        assert mask == [0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
        assert len(logprobs) == len(tokens) == len(mask)
        # Logprobs exist exactly where the rollout engine scored tokens.
        assert logprobs[4:6] == [-0.5, -0.1]
        assert logprobs[9] == -0.9
        assert all(lp is None for lp in logprobs[:4])

    def test_trainable_tokens_are_exactly_policy_selected(self) -> None:
        trajectory = self.make_multi_turn_trajectory()
        tokens, mask, _ = trajectory.token_stream()
        trainable = [t for t, m in zip(tokens, mask, strict=True) if m == 1]
        assert trainable == [10, 11, 30]
        assert trajectory.num_trainable_tokens == 3

    def test_eventless_trajectory_flattens_empty(self) -> None:
        trajectory = Trajectory(
            run_id="r",
            rollout_id="ro",
            task_id="t",
            policy_version="v0",
            terminal_state=TerminalState.ERROR,
            termination_reason="sandbox failed to start",
        )
        assert trajectory.token_stream() == ([], [], [])
