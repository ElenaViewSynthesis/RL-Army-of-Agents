"""Typed models for CI and quality tools."""

from pydantic import BaseModel, Field


class CiResult(BaseModel):
    success: bool = True
    error: str | None = None


class CiCommandInput(BaseModel):
    root: str = "."
    paths: list[str] = Field(default_factory=list)


class CommandOutput(CiResult):
    command: list[str]
    return_code: int
    output: str


class QualitySummaryOutput(CiResult):
    root: str
    checks: dict[str, bool]
    summary: str
