"""BudgetTracker enforcement tests (deterministic fake clock)."""

from trajectoryos.agents import BudgetTracker
from trajectoryos.schemas import BudgetSpec


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestBudgetTracker:
    def test_within_budget_no_violation(self) -> None:
        tracker = BudgetTracker(BudgetSpec(max_output_tokens=100, max_tool_calls={"run_shell": 2}))
        tracker.add_tokens(output_tokens=100)
        tracker.record_tool_call("run_shell")
        assert tracker.violation() is None

    def test_output_token_cap(self) -> None:
        tracker = BudgetTracker(BudgetSpec(max_output_tokens=10))
        tracker.add_tokens(output_tokens=11)
        violation = tracker.violation()
        assert violation is not None and "max_output_tokens" in violation

    def test_input_token_cap(self) -> None:
        tracker = BudgetTracker(BudgetSpec(max_input_tokens=5))
        tracker.add_tokens(input_tokens=6)
        violation = tracker.violation()
        assert violation is not None and "max_input_tokens" in violation

    def test_per_tool_cap_only_affects_that_tool(self) -> None:
        tracker = BudgetTracker(BudgetSpec(max_tool_calls={"run_shell": 1}))
        tracker.record_tool_call("read_file")
        tracker.record_tool_call("read_file")
        tracker.record_tool_call("run_shell")
        assert tracker.violation() is None
        tracker.record_tool_call("run_shell")
        violation = tracker.violation()
        assert violation is not None and "max_tool_calls[run_shell]" in violation

    def test_wallclock_cap(self) -> None:
        clock = FakeClock()
        tracker = BudgetTracker(BudgetSpec(max_wallclock_seconds=60), clock=clock)
        clock.advance(59)
        assert tracker.violation() is None
        clock.advance(2)
        violation = tracker.violation()
        assert violation is not None and "max_wallclock_seconds" in violation

    def test_cost_cap(self) -> None:
        tracker = BudgetTracker(BudgetSpec(max_estimated_cost_usd=0.10))
        tracker.add_cost(0.11)
        violation = tracker.violation()
        assert violation is not None and "max_estimated_cost_usd" in violation

    def test_unbounded_budget_never_violates(self) -> None:
        tracker = BudgetTracker(BudgetSpec())
        tracker.add_tokens(input_tokens=10**9, output_tokens=10**9)
        for _ in range(1000):
            tracker.record_tool_call("run_shell")
        assert tracker.violation() is None

    def test_cost_summary_reflects_usage(self) -> None:
        clock = FakeClock()
        tracker = BudgetTracker(BudgetSpec(), clock=clock)
        tracker.add_tokens(input_tokens=100, output_tokens=50)
        tracker.record_tool_call("edit_file")
        clock.advance(12.5)
        summary = tracker.cost_summary()
        assert summary.input_tokens == 100
        assert summary.output_tokens == 50
        assert summary.tool_calls == {"edit_file": 1}
        assert summary.wallclock_seconds == 12.5
