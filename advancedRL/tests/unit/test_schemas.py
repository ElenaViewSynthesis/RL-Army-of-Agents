"""Schema validation and serialization tests."""

import pytest
from pydantic import ValidationError
from trajectoryos.schemas import (
    BudgetSpec,
    CostSummary,
    EnvironmentRef,
    EventType,
    ModelSpec,
    Role,
    TaskSpec,
    TerminalState,
    Trajectory,
    TrajectoryEvent,
    VerifierRef,
    VerifierResult,
)


def make_task(**kwargs: object) -> TaskSpec:
    defaults: dict[str, object] = {
        "task_id": "task-001",
        "prompt": "Fix the failing test in repo X.",
        "environment": EnvironmentRef(name="docker_repo_sandbox"),
        "verifier": VerifierRef(name="clean_sandbox_unit_tests"),
    }
    defaults.update(kwargs)
    return TaskSpec(**defaults)


class TestBudgetSpec:
    def test_defaults_are_unbounded(self) -> None:
        budget = BudgetSpec()
        assert budget.max_input_tokens is None
        assert budget.max_output_tokens is None
        assert budget.max_tool_calls == {}
        assert budget.max_wallclock_seconds is None
        assert budget.max_gpu_seconds is None
        assert budget.max_estimated_cost_usd is None

    def test_rejects_non_positive_caps(self) -> None:
        with pytest.raises(ValidationError):
            BudgetSpec(max_input_tokens=0)
        with pytest.raises(ValidationError):
            BudgetSpec(max_wallclock_seconds=-1.0)

    def test_rejects_negative_tool_caps(self) -> None:
        with pytest.raises(ValidationError):
            BudgetSpec(max_tool_calls={"run_shell": -1})

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            BudgetSpec(max_total_tokens=100)  # type: ignore[call-arg]


class TestTaskSpec:
    def test_roundtrip_json(self) -> None:
        task = make_task(budget=BudgetSpec(max_output_tokens=2048, max_tool_calls={"edit": 10}))
        restored = TaskSpec.model_validate_json(task.model_dump_json())
        assert restored == task

    def test_rejects_empty_task_id(self) -> None:
        with pytest.raises(ValidationError):
            make_task(task_id="")


class TestTrajectoryEvent:
    def test_policy_event_with_aligned_fields(self) -> None:
        event = TrajectoryEvent(
            event_type=EventType.POLICY_OUTPUT,
            role=Role.ASSISTANT,
            token_ids=[1, 2, 3],
            rollout_logprobs=[-0.1, -0.5, -0.2],
            loss_mask=[0, 1, 1],
        )
        assert event.num_trainable_tokens == 2

    def test_token_ids_require_explicit_loss_mask(self) -> None:
        with pytest.raises(ValidationError, match="explicit loss_mask"):
            TrajectoryEvent(
                event_type=EventType.POLICY_OUTPUT,
                role=Role.ASSISTANT,
                token_ids=[1, 2],
            )

    def test_rejects_mask_length_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="loss_mask length"):
            TrajectoryEvent(
                event_type=EventType.POLICY_OUTPUT,
                role=Role.ASSISTANT,
                token_ids=[1, 2, 3],
                loss_mask=[1, 1],
            )

    def test_rejects_logprob_length_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="rollout_logprobs length"):
            TrajectoryEvent(
                event_type=EventType.POLICY_OUTPUT,
                role=Role.ASSISTANT,
                token_ids=[1, 2, 3],
                loss_mask=[1, 1, 1],
                rollout_logprobs=[-0.1],
            )

    def test_rejects_logprobs_without_tokens(self) -> None:
        with pytest.raises(ValidationError, match="without token_ids"):
            TrajectoryEvent(
                event_type=EventType.POLICY_OUTPUT,
                role=Role.ASSISTANT,
                rollout_logprobs=[-0.1],
            )

    def test_rejects_mask_without_tokens(self) -> None:
        with pytest.raises(ValidationError, match="without token_ids"):
            TrajectoryEvent(
                event_type=EventType.POLICY_OUTPUT,
                role=Role.ASSISTANT,
                loss_mask=[1],
            )

    def test_rejects_non_binary_mask(self) -> None:
        with pytest.raises(ValidationError, match="0 or 1"):
            TrajectoryEvent(
                event_type=EventType.POLICY_OUTPUT,
                role=Role.ASSISTANT,
                token_ids=[1, 2],
                loss_mask=[1, 2],
            )


class TestTrajectory:
    def test_minimal_failed_trajectory_is_representable(self) -> None:
        """Failed/truncated rollouts are first-class: state + reason recorded."""
        trajectory = Trajectory(
            run_id="run-1",
            rollout_id="ro-1",
            task_id="task-001",
            policy_version="qwen3-4b-base@step0",
            terminal_state=TerminalState.BUDGET_EXCEEDED,
            termination_reason="max_tool_calls[run_shell] exhausted",
        )
        assert trajectory.events == []
        assert trajectory.total_reward == 0.0

    def test_roundtrip_with_verifier_and_costs(self) -> None:
        trajectory = Trajectory(
            run_id="run-1",
            rollout_id="ro-2",
            task_id="task-001",
            policy_version="v0",
            terminal_state=TerminalState.COMPLETED,
            verifier_result=VerifierResult(passed=True, score=1.0),
            cost_summary=CostSummary(input_tokens=900, output_tokens=120),
            reward_components={"task_success": 1.0, "token_cost": -0.02},
            total_reward=0.98,
        )
        restored = Trajectory.model_validate_json(trajectory.model_dump_json())
        assert restored == trajectory


class TestModelSpec:
    def test_defaults(self) -> None:
        spec = ModelSpec(hf_repo="Qwen/Qwen3-4B-Base")
        assert spec.is_moe is False
        assert spec.max_context_tokens == 32768
