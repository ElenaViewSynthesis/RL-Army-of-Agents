from pydantic import ValidationError

from agent.models.git import (
    GitCommitInput,
    GitCommitOutput,
    GitLogInput,
    GitStatusOutput,
)


def test_git_log_input_bounds_max_count() -> None:
    model = GitLogInput(max_count=5)

    assert model.max_count == 5


def test_git_log_input_rejects_zero_max_count() -> None:
    try:
        GitLogInput(max_count=0)
    except ValidationError:
        return

    raise AssertionError("zero max_count should fail validation")


def test_git_commit_input_requires_message() -> None:
    try:
        GitCommitInput(message="")
    except ValidationError:
        return

    raise AssertionError("empty commit message should fail validation")


def test_git_status_output_shape() -> None:
    output = GitStatusOutput(
        repo=".",
        branch="main",
        is_clean=True,
        output="",
    )

    assert output.success is True
    assert output.branch == "main"


def test_git_commit_output_allows_missing_sha_on_failure() -> None:
    output = GitCommitOutput(
        repo=".",
        output="nothing to commit",
        success=False,
        error="nothing to commit",
    )

    assert output.commit_sha is None
    assert output.success is False
