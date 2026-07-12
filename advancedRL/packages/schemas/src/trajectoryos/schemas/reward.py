"""Reward result: every component logged independently, plus the composed total.

Quality-signal fields (``task_success`` .. ``format_compliance``) are in [0, 1].
Cost fields are normalized, non-negative, and may exceed 1.0 when a rollout ran
over budget. ``total_reward`` is computed by ``trajectoryos.rewards.compute_reward``.
"""

from pydantic import BaseModel, ConfigDict, Field


class RewardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_success: float = Field(ge=0.0, le=1.0)
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    format_compliance: float = Field(default=0.0, ge=0.0, le=1.0)
    token_cost: float = Field(default=0.0, ge=0.0)
    tool_cost: float = Field(default=0.0, ge=0.0)
    latency_cost: float = Field(default=0.0, ge=0.0)
    gpu_cost: float = Field(default=0.0, ge=0.0)
    retry_cost: float = Field(default=0.0, ge=0.0)
    safety_penalty: float = Field(default=0.0, ge=0.0)
    total_reward: float
