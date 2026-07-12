"""Docker sandbox: isolated task workspace in a container.

Drives the ``docker`` CLI directly (no daemon-API client dependency). Defaults
to ``--network none`` so agent code cannot reach the network, plus memory/CPU
caps. The task repository is copied in at creation, so the container never
mounts host paths read-write.
"""

import shlex
import subprocess
import time
from pathlib import Path

from trajectoryos.environments.base import ExecResult, SandboxError, validate_relative_path

_DOCKER_TIMEOUT = 60.0


def _docker(
    *args: str,
    input_text: str | None = None,
    timeout: float = _DOCKER_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class DockerSandbox:
    WORKDIR = "/workspace"

    def __init__(
        self,
        source_dir: str | Path,
        *,
        image: str = "python:3.12-slim",
        network: str = "none",
        memory: str = "1g",
        cpus: float = 1.0,
    ) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise SandboxError(f"source_dir does not exist: {source}")
        run = _docker(
            "run",
            "--detach",
            f"--network={network}",
            f"--memory={memory}",
            f"--cpus={cpus}",
            "--workdir",
            self.WORKDIR,
            image,
            "sleep",
            "infinity",
        )
        if run.returncode != 0:
            raise SandboxError(f"docker run failed: {run.stderr.strip()}")
        self._container = run.stdout.strip()
        # `source/.` copies directory contents into WORKDIR rather than nesting the dir.
        copy = _docker("cp", f"{source}/.", f"{self._container}:{self.WORKDIR}")
        if copy.returncode != 0:
            self.close()
            raise SandboxError(f"docker cp failed: {copy.stderr.strip()}")

    @staticmethod
    def is_available() -> bool:
        """True when the docker CLI exists and the daemon responds."""
        try:
            return _docker("info", timeout=15.0).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def read_file(self, path: str) -> str:
        relative = validate_relative_path(path)
        result = _docker("exec", self._container, "cat", f"{self.WORKDIR}/{relative}")
        if result.returncode != 0:
            raise SandboxError(f"read_file({path!r}) failed: {result.stderr.strip()}")
        return result.stdout

    def write_file(self, path: str, content: str) -> None:
        relative = validate_relative_path(path)
        target = f"{self.WORKDIR}/{relative}"
        parent = str(Path(target).parent).replace("\\", "/")
        script = f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(target)}"
        result = _docker(
            "exec", "--interactive", self._container, "sh", "-c", script, input_text=content
        )
        if result.returncode != 0:
            raise SandboxError(f"write_file({path!r}) failed: {result.stderr.strip()}")

    def exec(self, command: str, *, timeout_seconds: float = 120.0) -> ExecResult:
        start = time.monotonic()
        try:
            result = _docker(
                "exec",
                "--workdir",
                self.WORKDIR,
                self._container,
                "sh",
                "-c",
                command,
                timeout=timeout_seconds,
            )
            return ExecResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
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
        _docker("rm", "--force", self._container)

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _decode(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data
