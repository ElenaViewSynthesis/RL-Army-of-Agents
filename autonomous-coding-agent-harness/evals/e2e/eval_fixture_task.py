"""Metric-emitting e2e eval scaffold for the fixture repository.

This eval intentionally measures the fixture repo in its current state rather
than invoking a live LLM. Live agent success can be layered on top once model
credentials and runtime dependencies are configured.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _ROOT / "fixture_repo"


def run_fixture_tests() -> dict:
    """Run fixture pytest suite and emit a compact metric record."""
    command = [sys.executable, "-m", "pytest", str(_FIXTURE), "-q"]
    process = subprocess.run(
        command,
        cwd=_ROOT,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    return {
        "eval": "fixture_task_success",
        "command": command,
        "success": process.returncode == 0,
        "return_code": process.returncode,
        "stdout_tail": process.stdout[-1000:],
        "stderr_tail": process.stderr[-1000:],
    }


def main() -> None:
    print(json.dumps(run_fixture_tests(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
