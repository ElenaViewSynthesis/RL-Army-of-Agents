"""A tiny calculator module with a deliberate bug for the repo bug-fix task."""


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a + b  # BUG: should be a - b
