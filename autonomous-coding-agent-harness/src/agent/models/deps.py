"""Typed models for dependency tools."""

from pydantic import BaseModel, Field


class DependencyResult(BaseModel):
    success: bool = True
    error: str | None = None


class DependencyInput(BaseModel):
    root: str = "."


class DependencyInfo(BaseModel):
    name: str
    specifier: str = ""
    source: str


class ListDependenciesOutput(DependencyResult):
    root: str
    dependencies: list[DependencyInfo]


class CheckOutdatedOutput(DependencyResult):
    root: str
    packages: list[str]
    output: str


class ResolveImportInput(DependencyInput):
    module: str = Field(min_length=1)


class ResolveImportOutput(DependencyResult):
    module: str
    found: bool
    origin: str | None = None


class UnusedDepsOutput(DependencyResult):
    root: str
    dependencies: list[str]


class DependencyGraphOutput(DependencyResult):
    root: str
    edges: list[tuple[str, str]]


class VulnerabilityScanOutput(DependencyResult):
    root: str
    command: list[str]
    output: str


class AddDependencyInput(DependencyInput):
    requirement: str = Field(min_length=1)
    file_name: str = "requirements.txt"


class AddDependencyOutput(DependencyResult):
    path: str
    requirement: str
