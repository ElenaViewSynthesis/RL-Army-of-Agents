"""Reward-correctness tests: every component independently verifiable."""

from pathlib import Path

import pytest
from trajectoryos.core import load_config
from trajectoryos.rewards import (
    RewardInputs,
    RewardWeights,
    compute_reward,
    normalize_costs,
    reward_breakdown,
)
from trajectoryos.schemas import BudgetSpec, CostSummary

REPO_ROOT = Path(__file__).resolve().parents[2]

ALL_ONES = RewardWeights(
    success_weight=1.0,
    quality_weight=1.0,
    progress_weight=1.0,
    format_weight=1.0,
    token_penalty=1.0,
    tool_penalty=1.0,
    latency_penalty=1.0,
    gpu_penalty=1.0,
    retry_penalty=1.0,
    safety_weight=1.0,
)


class TestComposition:
    def test_pure_success_no_costs(self) -> None:
        result = compute_reward(RewardInputs(task_success=1.0), RewardWeights(success_weight=2.0))
        assert result.total_reward == pytest.approx(2.0)
        assert result.task_success == 1.0

    def test_breakdown_sums_to_total(self) -> None:
        inputs = RewardInputs(
            task_success=1.0,
            quality=0.7,
            progress=0.5,
            format_compliance=0.9,
            token_cost=0.4,
            tool_cost=0.3,
            latency_cost=0.2,
            gpu_cost=0.1,
            retry_count=2,
            safety_penalty=0.25,
        )
        breakdown = reward_breakdown(inputs, ALL_ONES)
        result = compute_reward(inputs, ALL_ONES)
        assert sum(breakdown.values()) == pytest.approx(result.total_reward)
        assert set(breakdown) == {
            "task_success",
            "quality",
            "progress",
            "format_compliance",
            "token_cost",
            "tool_cost",
            "latency_cost",
            "gpu_cost",
            "retry_cost",
            "safety_penalty",
        }

    @pytest.mark.parametrize(
        ("field", "delta_total"),
        [
            ("task_success", +0.5),
            ("quality", +0.5),
            ("progress", +0.5),
            ("format_compliance", +0.5),
            ("token_cost", -0.5),
            ("tool_cost", -0.5),
            ("latency_cost", -0.5),
            ("gpu_cost", -0.5),
            ("safety_penalty", -0.5),
        ],
    )
    def test_each_component_moves_total_independently(self, field: str, delta_total: float) -> None:
        base_inputs = RewardInputs(task_success=0.0)
        perturbed = base_inputs.model_copy(update={field: 0.5})
        base_total = compute_reward(base_inputs, ALL_ONES).total_reward
        new_total = compute_reward(perturbed, ALL_ONES).total_reward
        assert new_total - base_total == pytest.approx(delta_total)

    def test_retry_count_scales_linearly(self) -> None:
        weights = RewardWeights(retry_penalty=0.1)
        r0 = compute_reward(RewardInputs(task_success=1.0, retry_count=0), weights)
        r3 = compute_reward(RewardInputs(task_success=1.0, retry_count=3), weights)
        assert r0.total_reward - r3.total_reward == pytest.approx(0.3)
        assert r3.retry_cost == 3.0

    def test_zero_weight_silences_component(self) -> None:
        weights = RewardWeights(success_weight=1.0, token_penalty=0.0)
        result = compute_reward(RewardInputs(task_success=1.0, token_cost=100.0), weights)
        assert result.total_reward == pytest.approx(1.0)
        # ... but the raw component value is still logged.
        assert result.token_cost == 100.0

    def test_components_preserved_in_result(self) -> None:
        inputs = RewardInputs(task_success=1.0, token_cost=0.4, retry_count=1)
        result = compute_reward(inputs, ALL_ONES)
        assert result.token_cost == 0.4
        assert result.retry_cost == 1.0


class TestNormalization:
    def test_usage_over_caps(self) -> None:
        budget = BudgetSpec(
            max_input_tokens=1000,
            max_output_tokens=1000,
            max_tool_calls={"edit": 5, "run_shell": 5},
            max_wallclock_seconds=100.0,
            max_gpu_seconds=10.0,
        )
        cost = CostSummary(
            input_tokens=500,
            output_tokens=500,
            tool_calls={"edit": 2, "run_shell": 3},
            wallclock_seconds=50.0,
            gpu_seconds=5.0,
        )
        normalized = normalize_costs(cost, budget)
        assert normalized.token_cost == pytest.approx(0.5)
        assert normalized.tool_cost == pytest.approx(0.5)
        assert normalized.latency_cost == pytest.approx(0.5)
        assert normalized.gpu_cost == pytest.approx(0.5)

    def test_over_budget_exceeds_one(self) -> None:
        budget = BudgetSpec(max_output_tokens=100)
        cost = CostSummary(output_tokens=250)
        assert normalize_costs(cost, budget).token_cost == pytest.approx(2.5)

    def test_unbounded_dimensions_normalize_to_zero(self) -> None:
        normalized = normalize_costs(
            CostSummary(input_tokens=10_000, wallclock_seconds=999.0), BudgetSpec()
        )
        assert normalized.token_cost == 0.0
        assert normalized.tool_cost == 0.0
        assert normalized.latency_cost == 0.0
        assert normalized.gpu_cost == 0.0


class TestConfigIntegration:
    def test_default_weights_load_from_yaml(self) -> None:
        weights = load_config(
            REPO_ROOT / "configs" / "rewards" / "default.yaml", RewardWeights, env={}
        )
        assert weights.success_weight == 1.0
        assert weights.token_penalty == 0.05
        result = compute_reward(RewardInputs(task_success=1.0, token_cost=1.0), weights)
        assert result.total_reward == pytest.approx(0.95)
