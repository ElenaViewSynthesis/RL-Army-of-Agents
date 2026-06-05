"""Typed models for the first filesystem tool."""

from pydantic import BaseModel, Field


class ReadFileInput(BaseModel):
    """Input for reading a UTF-8 text file."""

    path: str = Field(min_length=1)
    encoding: str = "utf-8"


class ReadFileOutput(BaseModel):
    """Structured result returned by read_file."""

    path: str
    content: str
    size_bytes: int = Field(ge=0)
    success: bool = True
    error: str | None = None
