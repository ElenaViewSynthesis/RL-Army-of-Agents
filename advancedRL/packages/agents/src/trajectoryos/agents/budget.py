"""Budget tracking and enforcement for agent episodes.

The tracker accumulates usage; ``violation()`` returns a human-readable reason
string the moment any cap is exceeded (recorded verbatim as
``Trajectory.termination_reason``). Checked before/after every policy step and
tool call by the episode loop.
"""

import time
from collections import Counter
from collections.abc import Callable

from trajectoryos.schemas import BudgetSpec, CostSummary


class BudgetTracker:
    def __init__(self, budget: BudgetSpec, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._budget = budget
        self._clock = clock
        self._start = clock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_calls: Counter[str] = Counter()
        self._estimated_cost_usd = 0.0
        self._gpu_seconds = 0.0

    def add_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    def record_tool_call(self, tool_name: str) -> None:
        self._tool_calls[tool_name] += 1

    def add_cost(self, usd: float) -> None:
        self._estimated_cost_usd += usd

    def add_gpu_seconds(self, seconds: float) -> None:
        self._gpu_seconds += seconds

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._start

    def violation(self) -> str | None:
        """First exceeded cap as a reason string, or None while within budget."""
        budget = self._budget
        if budget.max_input_tokens is not None and self._input_tokens > budget.max_input_tokens:
            return f"max_input_tokens exceeded: {self._input_tokens} > {budget.max_input_tokens}"
        if budget.max_output_tokens is not None and self._output_tokens > budget.max_output_tokens:
            return f"max_output_tokens exceeded: {self._output_tokens} > {budget.max_output_tokens}"
        for tool, cap in budget.max_tool_calls.items():
            if self._tool_calls[tool] > cap:
                return f"max_tool_calls[{tool}] exceeded: {self._tool_calls[tool]} > {cap}"
        if (
            budget.max_wallclock_seconds is not None
            and self.elapsed_seconds > budget.max_wallclock_seconds
        ):
            return (
                f"max_wallclock_seconds exceeded: "
                f"{self.elapsed_seconds:.1f}s > {budget.max_wallclock_seconds}s"
            )
        if budget.max_gpu_seconds is not None and self._gpu_seconds > budget.max_gpu_seconds:
            return f"max_gpu_seconds exceeded: {self._gpu_seconds:.1f} > {budget.max_gpu_seconds}"
        if (
            budget.max_estimated_cost_usd is not None
            and self._estimated_cost_usd > budget.max_estimated_cost_usd
        ):
            return (
                f"max_estimated_cost_usd exceeded: "
                f"{self._estimated_cost_usd:.4f} > {budget.max_estimated_cost_usd}"
            )
        return None

    def cost_summary(self) -> CostSummary:
        return CostSummary(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            tool_calls=dict(self._tool_calls),
            wallclock_seconds=self.elapsed_seconds,
            gpu_seconds=self._gpu_seconds,
            estimated_cost_usd=self._estimated_cost_usd,
        )
