"""Backend discovery from source, without importing agent-controlled code."""

import ast
import os
import re
from pathlib import Path

DEVICE = os.environ.get("CUDA_AGENT_DEVICE", "cuda")


def _tree(workdir):
    return ast.parse((Path(workdir) / "model_new.py").read_text())


def detect_backend(workdir) -> str:
    """A literal module-level BACKEND overrides imports; legacy defaults to CUDA."""
    tree = _tree(workdir)
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == "BACKEND" for t in targets):
            value = ast.literal_eval(node.value)
            if value not in ("cuda", "triton"):
                raise ValueError('BACKEND must be "cuda" or "triton"')
            return value
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            modules += [f"{node.module}.{n.name}" for n in node.names]
        else:
            continue
        if any(m == "cuda_extension" or m.startswith("cuda_extension.") for m in modules):
            found.add("cuda")
        if any(m == "kernels.triton" or m.startswith("kernels.triton.") for m in modules):
            found.add("triton")
    if len(found) > 1:
        raise ValueError("Ambiguous backend imports; set module-level BACKEND")
    return next(iter(found), "cuda")


def agent_kernel_names(workdir, backend=None) -> list[str]:
    """Read authored CUDA globals or Triton jit functions, in stable name order."""
    root = Path(workdir)
    backend = backend or detect_backend(root)
    names = set()
    if backend == "cuda":
        for path in sorted((root / "kernels").glob("*.cu")):
            source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
            names.update(re.findall(
                r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?"
                r"(?:[\w:<>]+\s+)+?(\w+)\s*\(", source
            ))
    elif backend == "triton":
        for path in sorted((root / "kernels" / "triton").glob("*.py")):
            tree = ast.parse(path.read_text())
            triton_aliases, jit_aliases = {"triton"}, set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    triton_aliases.update(n.asname or n.name for n in node.names if n.name == "triton")
                if isinstance(node, ast.ImportFrom) and node.module == "triton":
                    jit_aliases.update(n.asname or n.name for n in node.names if n.name == "jit")
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    fn = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if (isinstance(fn, ast.Attribute) and fn.attr == "jit"
                            and isinstance(fn.value, ast.Name) and fn.value.id in triton_aliases
                            or isinstance(fn, ast.Name) and fn.id in jit_aliases):
                        names.add(node.name)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return sorted(names)
