"""Typed models for filesystem tools."""

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    success: bool = True
    error: str | None = None


class ReadFileInput(BaseModel):
    path: str = Field(min_length=1)
    encoding: str = "utf-8"


class ReadFileOutput(ToolResult):
    path: str
    content: str
    size_bytes: int = Field(ge=0)


class ReadFileRangeInput(BaseModel):
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    encoding: str = "utf-8"


class ReadFileRangeOutput(ToolResult):
    path: str
    lines: list[str]
    start_line: int
    end_line: int


class WriteFileInput(BaseModel):
    path: str = Field(min_length=1)
    content: str
    encoding: str = "utf-8"
    create_dirs: bool = False


class WriteFileOutput(ToolResult):
    path: str
    bytes_written: int = Field(ge=0)


class DirEntry(BaseModel):
    name: str
    is_dir: bool
    size_bytes: int = Field(ge=0)


class ListDirInput(BaseModel):
    path: str = "."
    recursive: bool = False


class ListDirOutput(ToolResult):
    path: str
    entries: list[DirEntry]
    count: int = Field(ge=0)


class SearchFilesInput(BaseModel):
    root: str = "."
    pattern: str = Field(min_length=1)
    max_results: int = Field(default=200, ge=1)


class SearchFilesOutput(ToolResult):
    root: str
    pattern: str
    matches: list[str]
    count: int = Field(ge=0)


class GrepInput(BaseModel):
    root: str = "."
    pattern: str = Field(min_length=1)
    file_glob: str = "*"
    max_results: int = Field(default=200, ge=1)
    case_sensitive: bool = True


class GrepMatch(BaseModel):
    file: str
    line_number: int = Field(ge=1)
    text: str


class GrepOutput(ToolResult):
    root: str
    pattern: str
    matches: list[GrepMatch]
    count: int = Field(ge=0)


class FileStatInput(BaseModel):
    path: str = Field(min_length=1)


class FileStatOutput(ToolResult):
    path: str
    size_bytes: int = Field(ge=0)
    is_file: bool
    is_dir: bool


class MakeDirInput(BaseModel):
    path: str = Field(min_length=1)
    parents: bool = True
    exist_ok: bool = True


class MakeDirOutput(ToolResult):
    path: str


class MoveInput(BaseModel):
    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    overwrite: bool = False


class MoveOutput(ToolResult):
    src: str
    dst: str


class DeleteInput(BaseModel):
    path: str = Field(min_length=1)
    recursive: bool = False


class DeleteOutput(ToolResult):
    path: str


class CopyInput(BaseModel):
    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    overwrite: bool = False


class CopyOutput(ToolResult):
    src: str
    dst: str
