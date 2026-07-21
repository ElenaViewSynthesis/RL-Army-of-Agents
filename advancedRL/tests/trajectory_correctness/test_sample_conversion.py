"""Trajectory → slime Sample conversion: token, mask and logprob correctness."""

import pytest
from trajectoryos.integrations.slime import (
    SampleConversionError,
    trajectory_to_sample_data,
)
from trajectoryos.schemas import (
    EventType,
    Role,
    TerminalState,
    Trajectory,
    TrajectoryEvent,
)


def context_event(tokens: list[int]) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_type=EventType.ENVIRONMENT_OBSERVATION,
        role=Role.ENVIRONMENT,
        token_ids=tokens,
        loss_mask=[0] * len(tokens),
        metadata={"kind": "ingested_context"},
    )


def policy_event(
    tokens: list[int], logprobs: list[float] | None = None, content: str = ""
) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_type=EventType.POLICY_OUTPUT,
        role=Role.ASSISTANT,
        content=content,
        token_ids=tokens,
        loss_mask=[1] * len(tokens),
        rollout_logprobs=logprobs,
    )


def make_trajectory(
    events: list[TrajectoryEvent],
    terminal_state: TerminalState = TerminalState.COMPLETED,
) -> Trajectory:
    return Trajectory(
        run_id="r1",
        rollout_id="ro1",
        task_id="t1",
        policy_version="v0",
        events=events,
        terminal_state=terminal_state,
    )


class TestConversion:
    def test_multi_turn_prompt_response_split(self) -> None:
        trajectory = make_trajectory(
            [
                context_event([1, 2, 3]),  # prompt region
                policy_event([10, 11], [-0.5, -0.1], content="calling tool"),
                context_event([20, 21]),  # tool observation inside response
                policy_event([30], [-0.9], content="submitting"),
            ]
        )
        sample = trajectory_to_sample_data(trajectory)

        assert sample.tokens == [1, 2, 3, 10, 11, 20, 21, 30]
        assert sample.response_length == 5
        assert sample.loss_mask == [1, 1, 0, 0, 1]
        # Exact sampled logprobs preserved; observation positions filled with 0.0.
        assert sample.rollout_log_probs == [-0.5, -0.1, 0.0, 0.0, -0.9]
        assert sample.status == "COMPLETED"
        assert sample.metadata["rollout_id"] == "ro1"

    def test_custom_masked_fill_value(self) -> None:
        trajectory = make_trajectory(
            [context_event([1]), policy_event([2], [-0.3]), context_event([3])]
        )
        sample = trajectory_to_sample_data(trajectory, masked_logprob_fill=-1e9)
        assert sample.rollout_log_probs == [-0.3, -1e9]

    def test_trainable_token_without_logprob_is_fatal(self) -> None:
        trajectory = make_trajectory([context_event([1, 2]), policy_event([10, 11], logprobs=None)])
        with pytest.raises(SampleConversionError, match="no rollout logprob"):
            trajectory_to_sample_data(trajectory)

    def test_tokenless_trajectory_is_fatal(self) -> None:
        trajectory = make_trajectory(
            [TrajectoryEvent(event_type=EventType.USER_MESSAGE, role=Role.USER, content="hi")]
        )
        with pytest.raises(SampleConversionError, match="no tokens"):
            trajectory_to_sample_data(trajectory)

    def test_no_assistant_tokens_is_fatal(self) -> None:
        trajectory = make_trajectory([context_event([1, 2, 3])])
        with pytest.raises(SampleConversionError, match="no assistant tokens"):
            trajectory_to_sample_data(trajectory)

    @pytest.mark.parametrize(
        ("terminal_state", "expected_status"),
        [
            (TerminalState.COMPLETED, "COMPLETED"),
            (TerminalState.TRUNCATED, "TRUNCATED"),
            (TerminalState.BUDGET_EXCEEDED, "TRUNCATED"),
            (TerminalState.ERROR, "ABORTED"),
            (TerminalState.ABORTED, "ABORTED"),
        ],
    )
    def test_status_mapping_never_discards_failures(
        self, terminal_state: TerminalState, expected_status: str
    ) -> None:
        """Truncated/failed rollouts convert too — they carry status, not silence."""
        trajectory = make_trajectory(
            [context_event([1]), policy_event([2], [-0.1])], terminal_state
        )
        sample = trajectory_to_sample_data(trajectory)
        assert sample.status == expected_status
        assert sample.metadata["terminal_state"] == terminal_state.value

    def test_prompt_and_response_texts(self) -> None:
        trajectory = make_trajectory(
            [
                TrajectoryEvent(
                    event_type=EventType.USER_MESSAGE, role=Role.USER, content="fix the bug"
                ),
                context_event([1]),
                policy_event([2], [-0.1], content="on it"),
            ]
        )
        sample = trajectory_to_sample_data(trajectory)
        assert sample.prompt == "fix the bug"
        assert sample.response == "on it"
