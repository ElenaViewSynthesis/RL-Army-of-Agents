from pathlib import Path

import pytest


@pytest.mark.integration
def test_fixture_repo_exists_for_long_horizon_task() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = root / "fixture_repo"

    assert (fixture / "calculator.py").exists()
    assert (fixture / "app.py").exists()
    assert (fixture / "test_calculator.py").exists()
