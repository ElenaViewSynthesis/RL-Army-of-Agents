"""Local-process sandbox: real execution in a temporary directory.

NOT ISOLATED — commands run as the current user on the host. Intended for unit
tests and local development on machines without a container runtime. Anything
security-relevant (untrusted model output, reward-hacking boundaries) must use
``DockerSandbox``.
"""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from trajectoryos.environments.base import ExecResult, SandboxError, validate_relative_path


class LocalProcessSandbox:
    def __init__(self, source_dir: str | Path) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise SandboxError(f"source_dir does not exist: {source}")
        self._root = Path(tempfile.mkdtemp(prefix="tos-sandbox-"))
        self._workspace = self._root / "workspace"
        shutil.copytree(source, self._workspace)
        self._closed = False

    def _resolve(self, path: str) -> Path:
        relative = validate_relative_path(path)
        return self._workspace / Path(*relative.parts)

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise SandboxError(f"no such file in workspace: {path!r}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def exec(self, command: str, *, timeout_seconds: float = 120.0) -> ExecResult:
        if self._closed:
            raise SandboxError("sandbox is closed")
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return ExecResult(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=-1,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr),
                duration_ms=(time.monotonic() - start) * 1000,
                timed_out=True,
            )

    def close(self) -> None:
        self._closed = True
        shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> "LocalProcessSandbox":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _decode(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data
