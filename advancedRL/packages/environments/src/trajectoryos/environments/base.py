"""Sandbox interface: the boundary between agents/verifiers and execution backends.

All paths are POSIX-style and relative to the sandbox workspace root; absolute
paths and ``..`` traversal are rejected by every implementation.
"""

from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class SandboxError(RuntimeError):
    """Raised for sandbox lifecycle/IO failures (not for non-zero exec exits)."""


class ExecResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class Sandbox(Protocol):
    """A workspace containing a copy of the task repository plus command execution."""

    def read_file(self, path: str) -> str: ...

    def write_file(self, path: str, content: str) -> None: ...

    def exec(self, command: str, *, timeout_seconds: float = 120.0) -> ExecResult: ...

    def close(self) -> None: ...


def validate_relative_path(path: str) -> PurePosixPath:
    """Reject absolute paths and parent-directory traversal."""
    pure = PurePosixPath(path.replace("\\", "/"))
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise SandboxError(f"path must be relative to the workspace and not escape it: {path!r}")
    if not pure.parts:
        raise SandboxError("empty path")
    return pure
