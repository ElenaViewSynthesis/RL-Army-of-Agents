"""LocalProcessSandbox + coding tools: real execution in a temp workspace."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from trajectoryos.agents import execute_tool
from trajectoryos.environments import LocalProcessSandbox, SandboxError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "buggy_calculator"


@pytest.fixture
def sandbox() -> Iterator[LocalProcessSandbox]:
    with LocalProcessSandbox(FIXTURE) as sandbox:
        yield sandbox


class TestLocalProcessSandbox:
    def test_workspace_is_a_copy(self, sandbox: LocalProcessSandbox) -> None:
        sandbox.write_file("calculator.py", "def add(a, b): return 0\n")
        original = (FIXTURE / "calculator.py").read_text(encoding="utf-8")
        assert "return 0" not in original

    def test_read_write_roundtrip(self, sandbox: LocalProcessSandbox) -> None:
        sandbox.write_file("notes/scratch.txt", "hello")
        assert sandbox.read_file("notes/scratch.txt") == "hello"

    def test_exec_captures_output_and_exit_code(self, sandbox: LocalProcessSandbox) -> None:
        result = sandbox.exec(f'"{sys.executable}" -c "print(41 + 1)"')
        assert result.ok
        assert "42" in result.stdout

    def test_exec_nonzero_exit(self, sandbox: LocalProcessSandbox) -> None:
        result = sandbox.exec(f'"{sys.executable}" -c "raise SystemExit(3)"')
        assert result.exit_code == 3
        assert not result.ok

    def test_exec_timeout(self, sandbox: LocalProcessSandbox) -> None:
        code = "import time; time.sleep(30)"
        result = sandbox.exec(f'"{sys.executable}" -c "{code}"', timeout_seconds=1.0)
        assert result.timed_out
        assert not result.ok

    @pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../b.txt"])
    def test_path_escape_rejected(self, sandbox: LocalProcessSandbox, path: str) -> None:
        with pytest.raises(SandboxError):
            sandbox.write_file(path, "x")


class TestTools:
    def test_read_file(self, sandbox: LocalProcessSandbox) -> None:
        result = execute_tool(sandbox, "read_file", {"path": "calculator.py"})
        assert result.ok
        assert "def subtract" in result.output

    def test_edit_file_unique_replacement(self, sandbox: LocalProcessSandbox) -> None:
        result = execute_tool(
            sandbox,
            "edit_file",
            {
                "path": "calculator.py",
                "old_string": "return a + b  # BUG: should be a - b",
                "new_string": "return a - b",
            },
        )
        assert result.ok
        assert result.modified_path == "calculator.py"
        assert result.modified_content is not None
        assert "a - b" in result.modified_content

    def test_edit_file_missing_string_fails(self, sandbox: LocalProcessSandbox) -> None:
        result = execute_tool(
            sandbox,
            "edit_file",
            {"path": "calculator.py", "old_string": "not present", "new_string": "x"},
        )
        assert not result.ok
        assert "not found" in result.output

    def test_edit_file_ambiguous_string_fails(self, sandbox: LocalProcessSandbox) -> None:
        result = execute_tool(
            sandbox,
            "edit_file",
            {"path": "calculator.py", "old_string": "a + b", "new_string": "a - b"},
        )
        assert not result.ok
        assert "must be unique" in result.output

    def test_run_shell_reports_failure(self, sandbox: LocalProcessSandbox) -> None:
        result = execute_tool(
            sandbox, "run_shell", {"command": f'"{sys.executable}" test_calculator.py'}
        )
        assert not result.ok  # bug still present
        assert "FAILED" in result.output

    def test_unknown_tool(self, sandbox: LocalProcessSandbox) -> None:
        result = execute_tool(sandbox, "format_disk", {})
        assert not result.ok
        assert "unknown tool" in result.output
