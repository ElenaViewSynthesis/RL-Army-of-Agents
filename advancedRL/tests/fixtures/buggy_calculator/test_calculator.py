"""Canonical tests for the bug-fix task.

Runnable with plain `python test_calculator.py` (exit code 0/1) so no test
framework needs to exist inside the sandbox image. The verifier protects this
file: agent modifications to it are rejected.
"""

import sys

from calculator import add, subtract


def test_add() -> None:
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract() -> None:
    assert subtract(5, 3) == 2
    assert subtract(0, 4) == -4


if __name__ == "__main__":
    try:
        test_add()
        test_subtract()
    except AssertionError as exc:
        print(f"FAILED: {exc!r}")
        sys.exit(1)
    print("OK: 2 tests passed")
