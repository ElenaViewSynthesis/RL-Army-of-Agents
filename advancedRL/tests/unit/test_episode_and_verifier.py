"""End-to-end Milestone 2 flow: scripted policy fixes a bug in one sandbox,
a clean second sandbox verifies it, reward hacking is rejected.

Uses LocalProcessSandbox (real file IO + real subprocess execution); only the
policy's decision sequence is scripted.
"""

import sys
from pathlib import Path

from trajectoryos.agents import PolicyTurn, ScriptedPolicy, ToolCall, run_episode
from trajectoryos.environments import LocalProcessSandbox
from trajectoryos.schemas import (
    BudgetSpec,
    EnvironmentRef,
    EventType,
    Role,
    TaskSpec,
    TerminalState,
    VerifierRef,
)
from trajectoryos.verifiers import CleanSandboxTestVerifier

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "buggy_calculator"
TEST_COMMAND = f'"{sys.executable}" test_calculator.py'

FIX_EDIT = ToolCall(
    tool_name="edit_file",
    arguments={
        "path": "calculator.py",
        "old_string": "return a + b  # BUG: should be a - b",
        "new_string": "return a - b",
    },
)


def make_task(budget: BudgetSpec | None = None) -> TaskSpec:
    return TaskSpec(
        task_id="bugfix-calculator-001",
        prompt="Issue: subtract(5, 3) returns 8 instead of 2. Fix the bug and make tests pass.",
        environment=EnvironmentRef(name="local_process", config={"source_dir": str(FIXTURE)}),
        verifier=VerifierRef(name="clean_sandbox_unit_tests"),
        budget=budget
        or BudgetSpec(max_tool_calls={"read_file": 5, "edit_file": 5, "run_shell": 5}),
    )


def make_verifier() -> CleanSandboxTestVerifier:
    return CleanSandboxTestVerifier(
        sandbox_factory=lambda: LocalProcessSandbox(FIXTURE),
        test_command=TEST_COMMAND,
        timeout_seconds=60.0,
    )


def fixing_policy() -> ScriptedPolicy:
    return ScriptedPolicy(
        [
            PolicyTurn(
                content="Reading the buggy module.",
                tool_call=ToolCall(tool_name="read_file", arguments={"path": "calculator.py"}),
                input_tokens_used=50,
                output_tokens_used=20,
            ),
            PolicyTurn(
                content="Fixing subtract().",
                tool_call=FIX_EDIT,
                input_tokens_used=80,
                output_tokens_used=40,
            ),
            PolicyTurn(
                content="Running the tests to confirm.",
                tool_call=ToolCall(tool_name="run_shell", arguments={"command": TEST_COMMAND}),
                input_tokens_used=90,
                output_tokens_used=25,
            ),
            PolicyTurn(content="Bug fixed; submitting.", output_tokens_used=10),
        ]
    )


class TestEpisode:
    def test_full_bugfix_episode_and_clean_verification(self) -> None:
        task = make_task()
        with LocalProcessSandbox(FIXTURE) as sandbox:
            result = run_episode(
                task, fixing_policy(), sandbox, run_id="run-t1", policy_version="scripted@m2"
            )

        trajectory = result.trajectory
        assert trajectory.terminal_state is TerminalState.COMPLETED
        assert trajectory.termination_reason is None
        assert result.patch.keys() == {"calculator.py"}

        # Every action was recorded: 1 user + 4 policy + 3 tool calls + 3 tool results.
        types = [e.event_type for e in trajectory.events]
        assert types.count(EventType.POLICY_OUTPUT) == 4
        assert types.count(EventType.TOOL_CALL) == 3
        assert types.count(EventType.TOOL_RESULT) == 3
        assert all(
            e.role is not Role.ASSISTANT or e.event_type != EventType.TOOL_RESULT
            for e in trajectory.events
        )

        # Costs were tracked.
        assert trajectory.cost_summary.output_tokens == 95
        assert trajectory.cost_summary.tool_calls == {
            "read_file": 1,
            "edit_file": 1,
            "run_shell": 1,
        }

        # Clean-sandbox verification passes with the patch...
        verifier_result = make_verifier().verify(result.patch)
        assert verifier_result.passed
        assert verifier_result.details["applied_paths"] == ["calculator.py"]
        # ...and fails without it (the fix was real, not an artifact of the agent sandbox).
        assert not make_verifier().verify({}).passed

    def test_tool_call_budget_terminates_episode(self) -> None:
        task = make_task(BudgetSpec(max_tool_calls={"read_file": 1}))
        read_forever = ScriptedPolicy(
            [
                PolicyTurn(
                    tool_call=ToolCall(tool_name="read_file", arguments={"path": "calculator.py"})
                )
                for _ in range(5)
            ]
        )
        with LocalProcessSandbox(FIXTURE) as sandbox:
            result = run_episode(
                task, read_forever, sandbox, run_id="run-t2", policy_version="scripted@m2"
            )
        trajectory = result.trajectory
        assert trajectory.terminal_state is TerminalState.BUDGET_EXCEEDED
        assert trajectory.termination_reason is not None
        assert "max_tool_calls[read_file]" in trajectory.termination_reason
        assert any(e.event_type is EventType.BUDGET_VIOLATION for e in trajectory.events)
        # The capped call was not executed: only the first read produced a result.
        assert sum(1 for e in trajectory.events if e.event_type is EventType.TOOL_RESULT) == 1

    def test_output_token_budget_terminates_episode(self) -> None:
        task = make_task(BudgetSpec(max_output_tokens=30))
        chatty = ScriptedPolicy(
            [
                PolicyTurn(
                    content="thinking a lot...",
                    output_tokens_used=31,
                    tool_call=ToolCall(tool_name="read_file", arguments={"path": "calculator.py"}),
                )
            ]
        )
        with LocalProcessSandbox(FIXTURE) as sandbox:
            result = run_episode(
                task, chatty, sandbox, run_id="run-t3", policy_version="scripted@m2"
            )
        assert result.trajectory.terminal_state is TerminalState.BUDGET_EXCEEDED
        assert result.trajectory.termination_reason is not None
        assert "max_output_tokens" in result.trajectory.termination_reason

    def test_max_steps_records_truncation(self) -> None:
        task = make_task()
        looping = ScriptedPolicy(
            [
                PolicyTurn(
                    tool_call=ToolCall(tool_name="read_file", arguments={"path": "calculator.py"})
                )
                for _ in range(10)
            ]
        )
        with LocalProcessSandbox(FIXTURE) as sandbox:
            result = run_episode(
                task,
                looping,
                sandbox,
                run_id="run-t4",
                policy_version="scripted@m2",
                max_steps=2,
            )
        assert result.trajectory.terminal_state is TerminalState.TRUNCATED
        assert result.trajectory.termination_reason is not None
        assert "max_steps" in result.trajectory.termination_reason


class TestRewardHackingPrevention:
    def test_patched_tests_are_rejected_and_pristine_tests_run(self) -> None:
        """An agent that rewrites the tests to always pass gains nothing."""
        hacked_tests = "print('OK: 2 tests passed')\n"
        verifier_result = make_verifier().verify({"test_calculator.py": hacked_tests})
        assert verifier_result.details["rejected_paths"] == ["test_calculator.py"]
        assert verifier_result.details["applied_paths"] == []
        # Pristine tests ran against the unfixed bug -> fail.
        assert not verifier_result.passed
        assert verifier_result.logs is not None and "FAILED" in verifier_result.logs

    def test_test_edit_alongside_real_fix_is_stripped_but_fix_counts(self) -> None:
        fixed = (
            (FIXTURE / "calculator.py")
            .read_text(encoding="utf-8")
            .replace("return a + b  # BUG: should be a - b", "return a - b")
        )
        verifier_result = make_verifier().verify(
            {"calculator.py": fixed, "test_calculator.py": "print('hacked')\n"}
        )
        assert verifier_result.passed
        assert verifier_result.details["rejected_paths"] == ["test_calculator.py"]
        assert verifier_result.details["applied_paths"] == ["calculator.py"]

    def test_protected_globs_cover_nested_tests_and_configs(self) -> None:
        verifier_result = make_verifier().verify(
            {
                "tests/test_deep.py": "x",
                "conftest.py": "x",
                "pytest.ini": "x",
                "pkg/utils_test.py": "x",
            }
        )
        assert verifier_result.details["applied_paths"] == []
        assert len(verifier_result.details["rejected_paths"]) == 4
