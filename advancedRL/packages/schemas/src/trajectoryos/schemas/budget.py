"""Budget specification for a task rollout.

``None`` means unbounded for that dimension. Enforcement happens in the agent
loop (Milestone 2); normalization against these caps happens in
``trajectoryos.rewards.normalize_costs``.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tool_calls: dict[str, int] = Field(
        default_factory=dict,
        description="Per-tool-name call caps; tools absent from the mapping are unbounded.",
    )
    max_wallclock_seconds: float | None = Field(default=None, gt=0)
    max_gpu_seconds: float | None = Field(default=None, gt=0)
    max_estimated_cost_usd: float | None = Field(default=None, gt=0)

    @field_validator("max_tool_calls")
    @classmethod
    def _non_negative_tool_caps(cls, v: dict[str, int]) -> dict[str, int]:
        for tool, cap in v.items():
            if cap < 0:
                raise ValueError(f"max_tool_calls[{tool!r}] must be >= 0, got {cap}")
        return v
