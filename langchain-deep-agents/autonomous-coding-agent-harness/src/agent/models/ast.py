"""Typed models for static-analysis tools."""

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    success: bool = True
    error: str | None = None


class ModuleInput(BaseModel):
    path: str = Field(min_length=1)


class SymbolInfo(BaseModel):
    name: str
    kind: str
    line: int = Field(ge=1)


class ParseModuleOutput(AnalysisResult):
    path: str
    node_count: int = Field(ge=0)
    syntax_ok: bool


class ListSymbolsOutput(AnalysisResult):
    path: str
    symbols: list[SymbolInfo]


class FindDefinitionInput(ModuleInput):
    name: str = Field(min_length=1)


class DefinitionLocation(BaseModel):
    name: str
    kind: str
    line: int = Field(ge=1)


class FindDefinitionOutput(AnalysisResult):
    path: str
    definitions: list[DefinitionLocation]


class FindReferencesInput(ModuleInput):
    name: str = Field(min_length=1)


class ReferenceLocation(BaseModel):
    path: str
    line: int = Field(ge=1)
    column: int = Field(ge=0)


class FindReferencesOutput(AnalysisResult):
    path: str
    references: list[ReferenceLocation]


class ImportInfo(BaseModel):
    module: str
    name: str | None = None
    line: int = Field(ge=1)


class ListImportsOutput(AnalysisResult):
    path: str
    imports: list[ImportInfo]


class ComplexityOutput(AnalysisResult):
    path: str
    functions: dict[str, int]


class DeadCodeOutput(AnalysisResult):
    path: str
    symbols: list[SymbolInfo]


class FunctionSignatureInput(ModuleInput):
    name: str = Field(min_length=1)


class FunctionSignatureOutput(AnalysisResult):
    path: str
    name: str
    signature: str | None = None


class UnusedImportsOutput(AnalysisResult):
    path: str
    imports: list[ImportInfo]
