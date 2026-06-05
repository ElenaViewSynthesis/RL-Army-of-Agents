from pydantic import ValidationError

from agent.models.fs import (
    CopyInput,
    GrepInput,
    ListDirOutput,
    ReadFileInput,
    ReadFileOutput,
    ReadFileRangeInput,
    WriteFileOutput,
)


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


def test_read_file_range_requires_positive_lines() -> None:
    try:
        ReadFileRangeInput(path="SPEC.md", start_line=0, end_line=1)
    except ValidationError:
        return

    raise AssertionError("line numbers should be 1-indexed")


def test_write_file_output_shape() -> None:
    output = WriteFileOutput(path="notes.md", bytes_written=12)

    assert output.success is True
    assert output.bytes_written == 12


def test_list_dir_output_counts_entries() -> None:
    output = ListDirOutput(path=".", entries=[], count=0)

    assert output.entries == []
    assert output.count == 0


def test_grep_input_defaults() -> None:
    model = GrepInput(pattern="Agent")

    assert model.root == "."
    assert model.file_glob == "*"
    assert model.case_sensitive is True


def test_copy_input_requires_source_and_destination() -> None:
    try:
        CopyInput(src="", dst="out.txt")
    except ValidationError:
        return

    raise AssertionError("empty source should fail validation")
