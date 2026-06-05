from pydantic import ValidationError

from agent.models.tool_io import ReadFileInput, ReadFileOutput


def test_read_file_input_accepts_path() -> None:
    model = ReadFileInput(path="SPEC.md")

    assert model.path == "SPEC.md"
    assert model.encoding == "utf-8"


def test_read_file_input_rejects_empty_path() -> None:
    try:
        ReadFileInput(path="")
    except ValidationError:
        return

    raise AssertionError("empty path should fail validation")


def test_read_file_output_success_shape() -> None:
    output = ReadFileOutput(path="SPEC.md", content="# SPEC", size_bytes=6)

    assert output.success is True
    assert output.error is None


def test_read_file_output_error_shape() -> None:
    output = ReadFileOutput(
        path="missing.md",
        content="",
        size_bytes=0,
        success=False,
        error="not found",
    )

    assert output.success is False
    assert output.error == "not found"


def test_read_file_output_rejects_negative_size() -> None:
    try:
        ReadFileOutput(path="SPEC.md", content="", size_bytes=-1)
    except ValidationError:
        return

    raise AssertionError("negative size should fail validation")
