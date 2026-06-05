"""MCP server for the git namespace."""

import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.models.git import (  # noqa: E402
    GitBlameInput,
    GitBlameLine,
    GitBlameOutput,
    GitBranchCreateInput,
    GitBranchCreateOutput,
    GitBranchListInput,
    GitBranchListOutput,
    GitCheckoutInput,
    GitCheckoutOutput,
    GitCommitInput,
    GitCommitOutput,
    GitCommitSummary,
    GitDiffInput,
    GitDiffOutput,
    GitListChangedFilesInput,
    GitListChangedFilesOutput,
    GitLogInput,
    GitLogOutput,
    GitShowCommitInput,
    GitShowCommitOutput,
    GitStashInput,
    GitStashOutput,
    GitStatusInput,
    GitStatusOutput,
    GitTagInput,
    GitTagOutput,
)

mcp = FastMCP("git")

_HEX = set("0123456789abcdef")


def _run(repo: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )


def _output(process: subprocess.CompletedProcess[str]) -> str:
    return (process.stdout + process.stderr).strip()


@mcp.tool()
def git_status(repo: str = ".") -> dict:
    """Return porcelain git status and current branch."""
    request = GitStatusInput(repo=repo)
    try:
        status = _run(request.repo, ["status", "--short"])
        branch = _run(request.repo, ["branch", "--show-current"])
        return GitStatusOutput(
            repo=request.repo,
            branch=branch.stdout.strip(),
            is_clean=status.stdout.strip() == "",
            output=_output(status),
            success=status.returncode == 0 and branch.returncode == 0,
            error=None if status.returncode == 0 else _output(status),
        ).model_dump()
    except Exception as exc:
        return GitStatusOutput(
            repo=request.repo,
            branch="",
            is_clean=False,
            output="",
            success=False,
            error=str(exc),
        ).model_dump()


@mcp.tool()
def git_diff(repo: str = ".", staged: bool = False, path: str | None = None) -> dict:
    """Return a git diff."""
    request = GitDiffInput(repo=repo, staged=staged, path=path)
    args = ["diff", "--cached" if request.staged else ""]
    args = [arg for arg in args if arg]
    if request.path:
        args.extend(["--", request.path])
    process = _run(request.repo, args)
    return GitDiffOutput(
        repo=request.repo,
        diff=process.stdout,
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_log(repo: str = ".", max_count: int = 10) -> dict:
    """Return recent commit summaries."""
    request = GitLogInput(repo=repo, max_count=max_count)
    process = _run(request.repo, ["log", f"--max-count={request.max_count}", "--format=%H%x00%s"])
    commits = []
    if process.returncode == 0:
        for line in process.stdout.splitlines():
            sha, _, message = line.partition("\x00")
            commits.append(GitCommitSummary(sha=sha, message=message))
    return GitLogOutput(
        repo=request.repo,
        commits=commits,
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_blame(
    repo: str = ".",
    path: str = "",
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Return porcelain blame information for a file."""
    request = GitBlameInput(repo=repo, path=path, start_line=start_line, end_line=end_line)
    args = ["blame", "--line-porcelain"]
    if request.start_line and request.end_line:
        args.extend(["-L", f"{request.start_line},{request.end_line}"])
    args.append(request.path)
    process = _run(request.repo, args)
    lines = []
    if process.returncode == 0:
        current_commit = ""
        current_line = 0
        for raw in process.stdout.splitlines():
            if raw and not raw.startswith("\t") and " " in raw:
                parts = raw.split()
                if (
                    len(parts) >= 3
                    and len(parts[0]) == 40
                    and all(char in _HEX for char in parts[0].lower())
                ):
                    current_commit = parts[0]
                    current_line = int(parts[2])
            elif raw.startswith("\t"):
                lines.append(
                    GitBlameLine(
                        commit=current_commit,
                        line_number=current_line,
                        text=raw[1:],
                    )
                )
    return GitBlameOutput(
        repo=request.repo,
        path=request.path,
        lines=lines,
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_branch_create(repo: str = ".", name: str = "", checkout: bool = True) -> dict:
    """Create a branch, optionally checking it out."""
    request = GitBranchCreateInput(repo=repo, name=name, checkout=checkout)
    args = ["checkout", "-b", request.name] if request.checkout else ["branch", request.name]
    process = _run(request.repo, args)
    return GitBranchCreateOutput(
        repo=request.repo,
        name=request.name,
        checked_out=request.checkout and process.returncode == 0,
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_branch_list(repo: str = ".") -> dict:
    """List local branches."""
    request = GitBranchListInput(repo=repo)
    process = _run(request.repo, ["branch", "--list"])
    branches = []
    current = None
    if process.returncode == 0:
        for line in process.stdout.splitlines():
            clean = line.strip()
            if clean.startswith("* "):
                current = clean[2:]
                branches.append(current)
            else:
                branches.append(clean)
    return GitBranchListOutput(
        repo=request.repo,
        branches=branches,
        current=current,
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_checkout(repo: str = ".", ref: str = "") -> dict:
    """Checkout a git ref."""
    request = GitCheckoutInput(repo=repo, ref=ref)
    process = _run(request.repo, ["checkout", request.ref])
    return GitCheckoutOutput(
        repo=request.repo,
        ref=request.ref,
        output=_output(process),
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_commit(repo: str = ".", message: str = "", all_changes: bool = False) -> dict:
    """Create a git commit."""
    request = GitCommitInput(repo=repo, message=message, all_changes=all_changes)
    if request.all_changes:
        add = _run(request.repo, ["add", "-A"])
        if add.returncode != 0:
            return GitCommitOutput(
                repo=request.repo,
                output=_output(add),
                success=False,
                error=_output(add),
            ).model_dump()
    process = _run(request.repo, ["commit", "-m", request.message])
    sha = None
    if process.returncode == 0:
        rev = _run(request.repo, ["rev-parse", "HEAD"])
        sha = rev.stdout.strip() if rev.returncode == 0 else None
    return GitCommitOutput(
        repo=request.repo,
        commit_sha=sha,
        output=_output(process),
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_stash(repo: str = ".", message: str | None = None) -> dict:
    """Stash current worktree changes."""
    request = GitStashInput(repo=repo, message=message)
    args = ["stash", "push"]
    if request.message:
        args.extend(["-m", request.message])
    process = _run(request.repo, args)
    return GitStashOutput(
        repo=request.repo,
        output=_output(process),
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_show_commit(repo: str = ".", ref: str = "HEAD") -> dict:
    """Show a commit."""
    request = GitShowCommitInput(repo=repo, ref=ref)
    process = _run(request.repo, ["show", "--stat", "--patch", request.ref])
    return GitShowCommitOutput(
        repo=request.repo,
        ref=request.ref,
        output=_output(process),
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_list_changed_files(repo: str = ".", staged: bool = False) -> dict:
    """List changed files."""
    request = GitListChangedFilesInput(repo=repo, staged=staged)
    args = ["diff", "--name-only", "--cached" if request.staged else ""]
    process = _run(request.repo, [arg for arg in args if arg])
    files = process.stdout.splitlines() if process.returncode == 0 else []
    return GitListChangedFilesOutput(
        repo=request.repo,
        files=files,
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


@mcp.tool()
def git_tag(
    repo: str = ".",
    name: str = "",
    ref: str = "HEAD",
    message: str | None = None,
) -> dict:
    """Create a lightweight or annotated tag."""
    request = GitTagInput(repo=repo, name=name, ref=ref, message=message)
    args = ["tag"]
    if request.message:
        args.extend(["-a", request.name, request.ref, "-m", request.message])
    else:
        args.extend([request.name, request.ref])
    process = _run(request.repo, args)
    return GitTagOutput(
        repo=request.repo,
        name=request.name,
        output=_output(process),
        success=process.returncode == 0,
        error=None if process.returncode == 0 else _output(process),
    ).model_dump()


if __name__ == "__main__":
    mcp.run()
