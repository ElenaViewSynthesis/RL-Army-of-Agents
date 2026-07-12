"""Docker sandbox isolation tests. Skipped automatically when no daemon is running.

Run with Docker Desktop (or any dockerd) up:  uv run pytest tests/integration -v
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from trajectoryos.environments import DockerSandbox, SandboxError
from trajectoryos.verifiers import CleanSandboxTestVerifier

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "buggy_calculator"

pytestmark = pytest.mark.skipif(
    not DockerSandbox.is_available(), reason="docker daemon not available"
)


@pytest.fixture
def sandbox() -> Iterator[DockerSandbox]:
    with DockerSandbox(FIXTURE) as sandbox:
        yield sandbox


class TestDockerSandbox:
    def test_repo_copied_into_workspace(self, sandbox: DockerSandbox) -> None:
        assert "def subtract" in sandbox.read_file("calculator.py")

    def test_exec_and_write_roundtrip(self, sandbox: DockerSandbox) -> None:
        sandbox.write_file("hello.py", "print('containerized')\n")
        result = sandbox.exec("python hello.py")
        assert result.ok
        assert "containerized" in result.stdout

    def test_network_is_disabled(self, sandbox: DockerSandbox) -> None:
        """Isolation: agent code cannot reach the network."""
        code = (
            "import socket, sys\n"
            "s = socket.socket()\n"
            "s.settimeout(5)\n"
            "try:\n"
            "    s.connect(('1.1.1.1', 80))\n"
            "except OSError:\n"
            "    sys.exit(42)\n"
            "sys.exit(0)\n"
        )
        sandbox.write_file("netcheck.py", code)
        result = sandbox.exec("python netcheck.py", timeout_seconds=30)
        assert result.exit_code == 42

    def test_agent_workspace_cannot_affect_clean_verifier_sandbox(
        self, sandbox: DockerSandbox
    ) -> None:
        """Reward-hacking boundary: tampering in the work sandbox is invisible
        to the fresh evaluation sandbox."""
        sandbox.write_file("test_calculator.py", "print('OK: 2 tests passed')\n")
        verifier = CleanSandboxTestVerifier(
            sandbox_factory=lambda: DockerSandbox(FIXTURE),
            test_command="python test_calculator.py",
            timeout_seconds=120.0,
        )
        result = verifier.verify({})  # no legitimate patch produced
        assert not result.passed  # pristine tests still see the bug

    def test_path_escape_rejected(self, sandbox: DockerSandbox) -> None:
        with pytest.raises(SandboxError):
            sandbox.read_file("../etc/passwd")
