"""MCP server for dependency inspection tools."""

import importlib.util
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.models.deps import (  # noqa: E402
    AddDependencyInput,
    AddDependencyOutput,
    CheckOutdatedOutput,
    DependencyGraphOutput,
    DependencyInfo,
    DependencyInput,
    ListDependenciesOutput,
    ResolveImportInput,
    ResolveImportOutput,
    UnusedDepsOutput,
    VulnerabilityScanOutput,
)

mcp = FastMCP("deps")


def _requirements(root: Path) -> list[DependencyInfo]:
    deps = []
    req = root / "requirements.txt"
    if req.exists():
        for line in req.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                name = clean.split("==")[0].split(">=")[0].split("<")[0]
                deps.append(DependencyInfo(name=name, specifier=clean.removeprefix(name), source="requirements.txt"))
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            clean = line.strip().strip('",')
            if ">=" in clean or "==" in clean:
                name = clean.split("==")[0].split(">=")[0].split("<")[0]
                deps.append(DependencyInfo(name=name, specifier=clean.removeprefix(name), source="pyproject.toml"))
    return deps


@mcp.tool()
def list_dependencies(root: str = ".") -> dict:
    """List dependencies from requirements.txt and pyproject.toml."""
    request = DependencyInput(root=root)
    try:
        return ListDependenciesOutput(root=request.root, dependencies=_requirements(Path(request.root))).model_dump()
    except Exception as exc:
        return ListDependenciesOutput(root=request.root, dependencies=[], success=False, error=str(exc)).model_dump()


@mcp.tool()
def check_outdated(root: str = ".") -> dict:
    """Run pip list --outdated."""
    request = DependencyInput(root=root)
    process = subprocess.run(["python", "-m", "pip", "list", "--outdated"], cwd=request.root, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=60, check=False)
    return CheckOutdatedOutput(root=request.root, packages=[], output=process.stdout + process.stderr, success=process.returncode == 0, error=None if process.returncode == 0 else process.stderr.strip()).model_dump()


@mcp.tool()
def resolve_import(module: str, root: str = ".") -> dict:
    """Resolve whether a Python module can be imported."""
    request = ResolveImportInput(root=root, module=module)
    spec = importlib.util.find_spec(request.module)
    return ResolveImportOutput(module=request.module, found=spec is not None, origin=spec.origin if spec else None).model_dump()


@mcp.tool()
def find_unused_deps(root: str = ".") -> dict:
    """Return dependencies that are not obvious imports in source files."""
    request = DependencyInput(root=root)
    deps = _requirements(Path(request.root))
    source_text = "\n".join(path.read_text(errors="ignore") for path in Path(request.root).rglob("*.py"))
    unused = [dep.name for dep in deps if f"import {dep.name.replace('-', '_')}" not in source_text]
    return UnusedDepsOutput(root=request.root, dependencies=unused).model_dump()


@mcp.tool()
def dependency_graph(root: str = ".") -> dict:
    """Return a simple project-to-dependency edge list."""
    request = DependencyInput(root=root)
    edges = [("project", dep.name) for dep in _requirements(Path(request.root))]
    return DependencyGraphOutput(root=request.root, edges=edges).model_dump()


@mcp.tool()
def vulnerability_scan(root: str = ".") -> dict:
    """Run pip-audit if available."""
    request = DependencyInput(root=root)
    command = ["python", "-m", "pip_audit"]
    process = subprocess.run(command, cwd=request.root, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=60, check=False)
    return VulnerabilityScanOutput(root=request.root, command=command, output=process.stdout + process.stderr, success=process.returncode == 0, error=None if process.returncode == 0 else process.stderr.strip()).model_dump()


@mcp.tool()
def add_dependency(requirement: str, root: str = ".", file_name: str = "requirements.txt") -> dict:
    """Append a requirement to a dependency file."""
    request = AddDependencyInput(root=root, requirement=requirement, file_name=file_name)
    path = Path(request.root) / request.file_name
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{request.requirement}\n")
    return AddDependencyOutput(path=str(path), requirement=request.requirement).model_dump()


if __name__ == "__main__":
    mcp.run()
