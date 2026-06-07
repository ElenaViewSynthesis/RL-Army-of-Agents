"""Typed models for test-runner tools."""

from pydantic import BaseModel, Field


class TestToolResult(BaseModel):
    success: bool = True
    error: str | None = None


class TestCommandInput(BaseModel):
    path: str = "."
    extra_args: list[str] = Field(default_factory=list)


class TestNodeInput(BaseModel):
    node_id: str = Field(min_length=1)


class TestResult(BaseModel):
    node_id: str
    outcome: str
    output: str = ""


class DiscoverTestsOutput(TestToolResult):
    path: str
    tests: list[str]
    count: int = Field(ge=0)


class TestRunOutput(TestToolResult):
    command: list[str]
    return_code: int
    output: str
    results: list[TestResult] = Field(default_factory=list)


class CoverageOutput(TestToolResult):
    command: list[str]
    output: str


class TestFailure(BaseModel):
    test_id: str
    error_text: str


class LastFailuresOutput(TestToolResult):
    failures: list[TestFailure]
