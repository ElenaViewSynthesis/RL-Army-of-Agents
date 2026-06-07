"""MCP server for Python AST analysis."""

import ast
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.models.ast import (  # noqa: E402
    ComplexityOutput,
    DeadCodeOutput,
    DefinitionLocation,
    FindDefinitionInput,
    FindDefinitionOutput,
    FindReferencesInput,
    FindReferencesOutput,
    FunctionSignatureInput,
    FunctionSignatureOutput,
    ImportInfo,
    ListImportsOutput,
    ListSymbolsOutput,
    ModuleInput,
    ParseModuleOutput,
    ReferenceLocation,
    SymbolInfo,
    UnusedImportsOutput,
)

mcp = FastMCP("ast")


def _tree(path: str) -> tuple[ast.AST, str]:
    text = Path(path).read_text(encoding="utf-8")
    return ast.parse(text), text


def _symbols(tree: ast.AST) -> list[SymbolInfo]:
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(SymbolInfo(name=node.name, kind="class", line=node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(SymbolInfo(name=node.name, kind="function", line=node.lineno))
    return sorted(symbols, key=lambda item: (item.line, item.name))


@mcp.tool()
def parse_module(path: str) -> dict:
    """Parse a Python module and return AST size metadata."""
    request = ModuleInput(path=path)
    try:
        tree, _ = _tree(request.path)
        return ParseModuleOutput(
            path=request.path,
            node_count=sum(1 for _ in ast.walk(tree)),
            syntax_ok=True,
        ).model_dump()
    except Exception as exc:
        return ParseModuleOutput(
            path=request.path,
            node_count=0,
            syntax_ok=False,
            success=False,
            error=str(exc),
        ).model_dump()


@mcp.tool()
def list_symbols(path: str) -> dict:
    """List top-level and nested classes/functions."""
    request = ModuleInput(path=path)
    try:
        tree, _ = _tree(request.path)
        return ListSymbolsOutput(path=request.path, symbols=_symbols(tree)).model_dump()
    except Exception as exc:
        return ListSymbolsOutput(path=request.path, symbols=[], success=False, error=str(exc)).model_dump()


@mcp.tool()
def find_definition(path: str, name: str) -> dict:
    """Find function or class definitions by name."""
    request = FindDefinitionInput(path=path, name=name)
    try:
        tree, _ = _tree(request.path)
        definitions = [
            DefinitionLocation(name=s.name, kind=s.kind, line=s.line)
            for s in _symbols(tree)
            if s.name == request.name
        ]
        return FindDefinitionOutput(path=request.path, definitions=definitions).model_dump()
    except Exception as exc:
        return FindDefinitionOutput(path=request.path, definitions=[], success=False, error=str(exc)).model_dump()


@mcp.tool()
def find_references(path: str, name: str) -> dict:
    """Find name references in a Python module."""
    request = FindReferencesInput(path=path, name=name)
    try:
        tree, _ = _tree(request.path)
        refs = [
            ReferenceLocation(path=request.path, line=node.lineno, column=node.col_offset)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == request.name
        ]
        return FindReferencesOutput(path=request.path, references=refs).model_dump()
    except Exception as exc:
        return FindReferencesOutput(path=request.path, references=[], success=False, error=str(exc)).model_dump()


@mcp.tool()
def list_imports(path: str) -> dict:
    """List imports in a Python module."""
    request = ModuleInput(path=path)
    try:
        tree, _ = _tree(request.path)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(ImportInfo(module=alias.name, line=node.lineno) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(ImportInfo(module=module, name=alias.name, line=node.lineno) for alias in node.names)
        return ListImportsOutput(path=request.path, imports=imports).model_dump()
    except Exception as exc:
        return ListImportsOutput(path=request.path, imports=[], success=False, error=str(exc)).model_dump()


@mcp.tool()
def compute_complexity(path: str) -> dict:
    """Compute a small cyclomatic-complexity approximation per function."""
    request = ModuleInput(path=path)
    try:
        tree, _ = _tree(request.path)
        functions = {}
        branch_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.BoolOp, ast.Match)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = 1 + sum(isinstance(child, branch_nodes) for child in ast.walk(node))
        return ComplexityOutput(path=request.path, functions=functions).model_dump()
    except Exception as exc:
        return ComplexityOutput(path=request.path, functions={}, success=False, error=str(exc)).model_dump()


@mcp.tool()
def detect_dead_code(path: str) -> dict:
    """Return private symbols as a minimal dead-code candidate heuristic."""
    request = ModuleInput(path=path)
    try:
        tree, _ = _tree(request.path)
        candidates = [s for s in _symbols(tree) if s.name.startswith("_") and not s.name.startswith("__")]
        return DeadCodeOutput(path=request.path, symbols=candidates).model_dump()
    except Exception as exc:
        return DeadCodeOutput(path=request.path, symbols=[], success=False, error=str(exc)).model_dump()


@mcp.tool()
def extract_function_signature(path: str, name: str) -> dict:
    """Return a simple function signature string."""
    request = FunctionSignatureInput(path=path, name=name)
    try:
        tree, _ = _tree(request.path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == request.name:
                args = [arg.arg for arg in node.args.args]
                return FunctionSignatureOutput(
                    path=request.path,
                    name=request.name,
                    signature=f"{request.name}({', '.join(args)})",
                ).model_dump()
        return FunctionSignatureOutput(path=request.path, name=request.name, signature=None).model_dump()
    except Exception as exc:
        return FunctionSignatureOutput(path=request.path, name=request.name, success=False, error=str(exc)).model_dump()


@mcp.tool()
def find_unused_imports(path: str) -> dict:
    """Find imports whose local name is not referenced elsewhere."""
    request = ModuleInput(path=path)
    try:
        tree, _ = _tree(request.path)
        imports = ListImportsOutput.model_validate(list_imports(request.path)).imports
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        unused = [imp for imp in imports if (imp.name or imp.module.split(".")[0]) not in used]
        return UnusedImportsOutput(path=request.path, imports=unused).model_dump()
    except Exception as exc:
        return UnusedImportsOutput(path=request.path, imports=[], success=False, error=str(exc)).model_dump()


if __name__ == "__main__":
    mcp.run()
