"""MCP server for test-running tools."""

import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.models.test import (  # noqa: E402
    CoverageOutput,
    DiscoverTestsOutput,
    LastFailuresOutput,
    TestFailure,
    TestNodeInput,
    TestResult,
    TestRunOutput,
)

mcp = FastMCP("test")

_LAST_FAILURES: list[TestFailure] = []


def _run(args: list[str], cwd: str = ".") -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=120, check=False)


def _parse_failures(output: str) -> list[TestFailure]:
    failures = []
    for line in output.splitlines():
        if line.startswith("FAILED "):
            test_id = line.split(" ", 1)[1].split(" - ", 1)[0]
            failures.append(TestFailure(test_id=test_id, error_text=line))
    return failures


@mcp.tool()
def discover_tests(path: str = ".", extra_args: list[str] | None = None) -> dict:
    """Discover pytest node ids without running tests."""
    args = ["pytest", "--collect-only", "-q", path, *(extra_args or [])]
    process = _run(args)
    tests = [line.strip() for line in process.stdout.splitlines() if "::" in line]
    return DiscoverTestsOutput(
        path=path,
        tests=tests,
        count=len(tests),
        success=process.returncode in (0, 5),
        error=None if process.returncode in (0, 5) else process.stderr.strip(),
    ).model_dump()


@mcp.tool()
def run_test_file(path: str, extra_args: list[str] | None = None) -> dict:
    """Run a pytest file."""
    return run_suite(path=path, extra_args=extra_args)


@mcp.tool()
def run_test_node(node_id: str) -> dict:
    """Run a single pytest node id."""
    request = TestNodeInput(node_id=node_id)
    return run_suite(path=request.node_id, extra_args=None)


@mcp.tool()
def run_suite(path: str = ".", extra_args: list[str] | None = None) -> dict:
    """Run pytest and return command output."""
    global _LAST_FAILURES
    command = ["pytest", path, *(extra_args or [])]
    process = _run(command)
    output = process.stdout + process.stderr
    _LAST_FAILURES = _parse_failures(output)
    result = TestResult(node_id=path, outcome="passed" if process.returncode == 0 else "failed", output=output[-2000:])
    return TestRunOutput(command=command, return_code=process.returncode, output=output, results=[result], success=process.returncode == 0, error=None if process.returncode == 0 else output[-500:]).model_dump()


@mcp.tool()
def coverage_report(path: str = ".") -> dict:
    """Run pytest with coverage if pytest-cov is installed."""
    command = ["pytest", "--cov", path]
    process = _run(command)
    return CoverageOutput(command=command, output=process.stdout + process.stderr, success=process.returncode == 0, error=None if process.returncode == 0 else process.stderr.strip()).model_dump()


@mcp.tool()
def coverage_diff(path: str = ".") -> dict:
    """Return a minimal coverage diff placeholder based on current coverage output."""
    return coverage_report(path)


@mcp.tool()
def last_failures() -> dict:
    """Return failures captured by the last test run."""
    return LastFailuresOutput(failures=_LAST_FAILURES).model_dump()


@mcp.tool()
def rerun_failed() -> dict:
    """Rerun the last failed pytest node ids."""
    if not _LAST_FAILURES:
        return TestRunOutput(command=[], return_code=0, output="no previous failures", results=[]).model_dump()
    return run_suite(path=_LAST_FAILURES[0].test_id, extra_args=None)


if __name__ == "__main__":
    mcp.run()
