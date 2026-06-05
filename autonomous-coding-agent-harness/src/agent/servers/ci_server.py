"""MCP server for CI and quality tools."""

import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.models.ci import CiCommandInput, CommandOutput, QualitySummaryOutput  # noqa: E402

mcp = FastMCP("ci")


def _run(command: list[str], root: str) -> CommandOutput:
    process = subprocess.run(command, cwd=root, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=120, check=False)
    return CommandOutput(command=command, return_code=process.returncode, output=process.stdout + process.stderr, success=process.returncode == 0, error=None if process.returncode == 0 else (process.stderr or process.stdout)[-500:])


@mcp.tool()
def run_linter(root: str = ".", paths: list[str] | None = None) -> dict:
    """Run ruff lint if available."""
    request = CiCommandInput(root=root, paths=paths or ["."])
    return _run(["python", "-m", "ruff", "check", *request.paths], request.root).model_dump()


@mcp.tool()
def run_formatter(root: str = ".", paths: list[str] | None = None) -> dict:
    """Run ruff format check if available."""
    request = CiCommandInput(root=root, paths=paths or ["."])
    return _run(["python", "-m", "ruff", "format", "--check", *request.paths], request.root).model_dump()


@mcp.tool()
def run_type_check(root: str = ".", paths: list[str] | None = None) -> dict:
    """Run mypy if available."""
    request = CiCommandInput(root=root, paths=paths or ["."])
    return _run(["python", "-m", "mypy", *request.paths], request.root).model_dump()


@mcp.tool()
def build_check(root: str = ".", paths: list[str] | None = None) -> dict:
    """Run Python compileall as a minimal build check."""
    request = CiCommandInput(root=root, paths=paths or ["src"])
    return _run(["python", "-m", "compileall", *request.paths], request.root).model_dump()


@mcp.tool()
def pre_commit_run(root: str = ".", paths: list[str] | None = None) -> dict:
    """Run pre-commit if available."""
    request = CiCommandInput(root=root, paths=paths or [])
    command = ["python", "-m", "pre_commit", "run", "--all-files", *request.paths]
    return _run(command, request.root).model_dump()


@mcp.tool()
def run_security_scan(root: str = ".", paths: list[str] | None = None) -> dict:
    """Run bandit if available."""
    request = CiCommandInput(root=root, paths=paths or ["src"])
    return _run(["python", "-m", "bandit", "-r", *request.paths], request.root).model_dump()


@mcp.tool()
def summarize_quality(root: str = ".", paths: list[str] | None = None) -> dict:
    """Summarize lightweight local quality checks."""
    request = CiCommandInput(root=root, paths=paths or ["src"])
    build = build_check(request.root, request.paths)
    build_ok = CommandOutput.model_validate(build).success
    return QualitySummaryOutput(root=request.root, checks={"build_check": build_ok}, summary="build_check passed" if build_ok else "build_check failed", success=build_ok).model_dump()


if __name__ == "__main__":
    mcp.run()
