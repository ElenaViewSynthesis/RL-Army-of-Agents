"""Composable cost-aware reward.

reward =  success_weight  * task_success
        + quality_weight  * quality
        + progress_weight * progress
        + format_weight   * format_compliance
        - token_penalty   * normalized_token_cost
        - tool_penalty    * normalized_tool_cost
        - latency_penalty * normalized_latency
        - gpu_penalty     * normalized_gpu_seconds
        - retry_penalty   * retry_count
        - safety_weight   * safety_penalty

Every component is computed and returned independently (``reward_breakdown``)
so it can be logged and tested in isolation; the breakdown sums exactly to
``total_reward``.
"""

from pydantic import BaseModel, ConfigDict, Field
from trajectoryos.schemas import BudgetSpec, CostSummary, RewardResult


class RewardWeights(BaseModel):
    """Weights for the composite reward; loadable from ``configs/rewards/*.yaml``."""

    model_config = ConfigDict(extra="forbid")

    success_weight: float = Field(default=1.0, ge=0)
    quality_weight: float = Field(default=0.0, ge=0)
    progress_weight: float = Field(default=0.0, ge=0)
    format_weight: float = Field(default=0.0, ge=0)
    token_penalty: float = Field(default=0.0, ge=0)
    tool_penalty: float = Field(default=0.0, ge=0)
    latency_penalty: float = Field(default=0.0, ge=0)
    gpu_penalty: float = Field(default=0.0, ge=0)
    retry_penalty: float = Field(default=0.0, ge=0)
    safety_weight: float = Field(default=1.0, ge=0)


class RewardInputs(BaseModel):
    """Raw measurements for one trajectory, before weighting.

    Quality signals are in [0, 1]. Cost fields are *normalized* (see
    ``normalize_costs``) and may exceed 1.0 when the rollout ran over budget.
    """

    model_config = ConfigDict(extra="forbid")

    task_success: float = Field(ge=0.0, le=1.0)
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    format_compliance: float = Field(default=0.0, ge=0.0, le=1.0)
    token_cost: float = Field(default=0.0, ge=0.0)
    tool_cost: float = Field(default=0.0, ge=0.0)
    latency_cost: float = Field(default=0.0, ge=0.0)
    gpu_cost: float = Field(default=0.0, ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    safety_penalty: float = Field(default=0.0, ge=0.0)


class NormalizedCosts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_cost: float = Field(ge=0.0)
    tool_cost: float = Field(ge=0.0)
    latency_cost: float = Field(ge=0.0)
    gpu_cost: float = Field(ge=0.0)


def normalize_costs(cost: CostSummary, budget: BudgetSpec) -> NormalizedCosts:
    """Normalize raw usage against budget caps: usage / cap.

    Values may exceed 1.0 when a rollout ran over budget (penalties keep
    growing rather than saturating). Dimensions with no cap normalize to 0.0 —
    an unbounded budget expresses "don't penalize this".
    """
    token_cap = (budget.max_input_tokens or 0) + (budget.max_output_tokens or 0)
    token_used = cost.input_tokens + cost.output_tokens
    token_cost = token_used / token_cap if token_cap else 0.0

    tool_cap = sum(budget.max_tool_calls.values())
    tool_used = sum(cost.tool_calls.values())
    tool_cost = tool_used / tool_cap if tool_cap else 0.0

    latency_cost = (
        cost.wallclock_seconds / budget.max_wallclock_seconds
        if budget.max_wallclock_seconds
        else 0.0
    )
    gpu_cost = cost.gpu_seconds / budget.max_gpu_seconds if budget.max_gpu_seconds else 0.0

    return NormalizedCosts(
        token_cost=token_cost,
        tool_cost=tool_cost,
        latency_cost=latency_cost,
        gpu_cost=gpu_cost,
    )


def reward_breakdown(inputs: RewardInputs, weights: RewardWeights) -> dict[str, float]:
    """Signed, weighted contribution of every component. Sums to the total reward."""
    return {
        "task_success": weights.success_weight * inputs.task_success,
        "quality": weights.quality_weight * inputs.quality,
        "progress": weights.progress_weight * inputs.progress,
        "format_compliance": weights.format_weight * inputs.format_compliance,
        "token_cost": -weights.token_penalty * inputs.token_cost,
        "tool_cost": -weights.tool_penalty * inputs.tool_cost,
        "latency_cost": -weights.latency_penalty * inputs.latency_cost,
        "gpu_cost": -weights.gpu_penalty * inputs.gpu_cost,
        "retry_cost": -weights.retry_penalty * inputs.retry_count,
        "safety_penalty": -weights.safety_weight * inputs.safety_penalty,
    }


def compute_reward(inputs: RewardInputs, weights: RewardWeights) -> RewardResult:
    """Compose the total reward; raw component values are preserved in the result."""
    breakdown = reward_breakdown(inputs, weights)
    return RewardResult(
        task_success=inputs.task_success,
        quality=inputs.quality,
        progress=inputs.progress,
        format_compliance=inputs.format_compliance,
        token_cost=inputs.token_cost,
        tool_cost=inputs.tool_cost,
        latency_cost=inputs.latency_cost,
        gpu_cost=inputs.gpu_cost,
        retry_cost=float(inputs.retry_count),
        safety_penalty=inputs.safety_penalty,
        total_reward=sum(breakdown.values()),
    )
