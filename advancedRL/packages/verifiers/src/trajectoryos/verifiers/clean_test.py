"""Clean-sandbox unit-test verifier.

Anti-reward-hacking design:

1. Evaluation happens in a **fresh sandbox built from the pristine task source**,
   never in the sandbox the agent worked in. Whatever the agent did to its own
   workspace (editing tests, poisoning caches, faking output) cannot leak here.
2. The agent's patch is applied file-by-file, and any file matching a protected
   glob (tests, test configs) is **rejected and logged** — the canonical tests
   always run exactly as written.
3. Success is the test command's exit code in that clean sandbox, nothing else.
"""

from collections.abc import Callable, Mapping, Sequence
from fnmatch import fnmatch
from pathlib import PurePosixPath

from trajectoryos.environments import Sandbox
from trajectoryos.schemas import VerifierResult

DEFAULT_PROTECTED_GLOBS: tuple[str, ...] = (
    "test_*.py",
    "*_test.py",
    "tests/*",
    "tests/**/*",
    "conftest.py",
    "pytest.ini",
    "setup.cfg",
    "pyproject.toml",
)


def _is_protected(path: str, globs: Sequence[str]) -> bool:
    posix = str(PurePosixPath(path.replace("\\", "/")))
    name = PurePosixPath(posix).name
    return any(fnmatch(posix, g) or fnmatch(name, g) for g in globs)


class CleanSandboxTestVerifier:
    def __init__(
        self,
        sandbox_factory: Callable[[], Sandbox],
        test_command: str,
        *,
        protected_globs: Sequence[str] = DEFAULT_PROTECTED_GLOBS,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._sandbox_factory = sandbox_factory
        self._test_command = test_command
        self._protected_globs = tuple(protected_globs)
        self._timeout_seconds = timeout_seconds

    def verify(self, patch: Mapping[str, str]) -> VerifierResult:
        """Apply ``patch`` (path -> full new content) to a clean sandbox and run tests."""
        rejected = sorted(p for p in patch if _is_protected(p, self._protected_globs))
        applied = sorted(p for p in patch if p not in set(rejected))

        sandbox = self._sandbox_factory()
        try:
            for path in applied:
                sandbox.write_file(path, patch[path])
            result = sandbox.exec(self._test_command, timeout_seconds=self._timeout_seconds)
        finally:
            sandbox.close()

        passed = result.ok
        return VerifierResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "applied_paths": applied,
                "rejected_paths": rejected,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "test_command": self._test_command,
            },
            logs=f"{result.stdout}\n{result.stderr}".strip(),
        )
