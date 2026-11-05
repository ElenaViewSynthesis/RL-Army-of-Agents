"""Backend registry and discovery from source, without importing agent-controlled code.

Each backend is a BackendSpec: how model_new.py selects it, where its authored
kernels live, how their names are discovered (AST or regex only; agent code is
never imported here), whether vendor-library time counts toward provenance,
and how the trusted harness compiles and warms it up. This module must stay
importable without Torch.
"""

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEVICE = os.environ.get("CUDA_AGENT_DEVICE", "cuda")


@dataclass(frozen=True)
class BackendSpec:
    name: str
    display: str                       # human-readable name for harness messages
    import_prefixes: tuple[str, ...]   # importing one of these modules selects the backend
    kernel_glob: str                   # authored sources, relative to the task workdir
    decorators: tuple[str, ...]        # fully-qualified kernel decorators; () = CUDA regex
    compile_mode: str                  # "extension" (nvcc -> .so) or "jit" (one forward)
    kernel_package: str | None         # imported by the JIT compile path
    library_allowed: bool              # cuBLAS/CUTLASS/cuDNN time counts toward provenance
    profile_warmup_iters: int
    cache_env: tuple[tuple[str, str], ...] = ()  # (ENV_VAR, subdirectory of <workdir>/.cache)
    distributions: tuple[str, ...] = ()          # pip distributions the backend needs
    skill_doc: str = "SKILL.md"
    template: str = "model_new.py"


BACKENDS = {
    "cuda": BackendSpec(
        name="cuda", display="CUDA", import_prefixes=("cuda_extension",),
        kernel_glob="kernels/*.cu", decorators=(), compile_mode="extension",
        kernel_package=None, library_allowed=True, profile_warmup_iters=5,
        distributions=("torch",)),
    "triton": BackendSpec(
        # TRITON_CACHE_DIR is set for every backend: torch.compile emits Triton too.
        name="triton", display="Triton", import_prefixes=("kernels.triton",),
        kernel_glob="kernels/triton/*.py", decorators=("triton.jit",),
        compile_mode="jit", kernel_package="kernels.triton", library_allowed=False,
        profile_warmup_iters=20, cache_env=(("TRITON_CACHE_DIR", "triton"),),
        distributions=("torch", "triton"), skill_doc="SKILL_TRITON.md",
        template="model_new_triton.py"),
    "tilelang": BackendSpec(
        # TileLang emits `<prim_func name>_kernel` as the device symbol. The
        # newer style decorates the kernel function itself with @tilelang.jit.
        name="tilelang", display="TileLang", import_prefixes=("kernels.tilelang",),
        kernel_glob="kernels/tilelang/*.py",
        decorators=("tilelang.language.prim_func", "tilelang.cuda.language.prim_func",
                    "tilelang.jit"),
        compile_mode="jit", kernel_package="kernels.tilelang", library_allowed=False,
        profile_warmup_iters=20, cache_env=(("TILELANG_CACHE_DIR", "tilelang"),),
        distributions=("torch", "tilelang"), skill_doc="SKILL_TILELANG.md",
        template="model_new_tilelang.py"),
    "cutile": BackendSpec(
        # NVIDIA CUDA Tile (cuda.tile), compiled by tileiras on first launch.
        # Its device symbol naming is undocumented; verify on hardware.
        name="cutile", display="cuTile", import_prefixes=("kernels.cutile",),
        kernel_glob="kernels/cutile/*.py", decorators=("cuda.tile.kernel",),
        compile_mode="jit", kernel_package="kernels.cutile", library_allowed=False,
        profile_warmup_iters=20, distributions=("torch", "cuda-tile"),
        skill_doc="SKILL_CUTILE.md", template="model_new_cutile.py"),
}
DEFAULT_BACKEND = "cuda"
BACKEND_CHOICES = ("auto", *BACKENDS)


def get_spec(backend) -> BackendSpec:
    try:
        return BACKENDS[backend]
    except KeyError:
        raise ValueError(f"Unsupported backend: {backend}") from None


def available_backends() -> dict:
    """{name: None if its packages are installed, else a reason}. Imports nothing."""
    from importlib.metadata import PackageNotFoundError, distribution

    def missing(dist):
        try:
            distribution(dist)
            return False
        except PackageNotFoundError:
            return True

    result = {}
    for name, spec in BACKENDS.items():
        absent = [d for d in spec.distributions if missing(d)]
        result[name] = f"package not installed: {', '.join(absent)}" if absent else None
    return result


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
            if value not in BACKENDS:
                raise ValueError("BACKEND must be one of: " + ", ".join(f'"{n}"' for n in BACKENDS))
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
        for spec in BACKENDS.values():
            if any(m == p or m.startswith(p + ".") for m in modules for p in spec.import_prefixes):
                found.add(spec.name)
    if len(found) > 1:
        raise ValueError("Ambiguous backend imports; set module-level BACKEND")
    return next(iter(found), DEFAULT_BACKEND)


_CUDA_GLOBAL = re.compile(
    r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?"
    r"(?:[\w:<>]+\s+)+?(\w+)\s*\("
)


def _cuda_globals(paths) -> set[str]:
    names = set()
    for path in paths:
        source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        names.update(_CUDA_GLOBAL.findall(source))
    return names


def _dotted(expr):
    """['a', 'b', 'c'] for the expression a.b.c, or None for anything else."""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None
    parts.append(expr.id)
    return parts[::-1]


def decorated_functions(source, decorators) -> set[str]:
    """Names of functions whose decorator resolves to one of `decorators`.

    Top-level `import x.y as z` / `from x import y as z` aliases are resolved,
    and called decorators (`@jit()`) count. A decorator spelled with the
    canonical dotted name (e.g. `triton.jit`) counts even without an import.
    """
    tree = ast.parse(source)
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.asname:
                    aliases[n.asname] = n.name
                else:
                    base = n.name.split(".")[0]
                    aliases[base] = base
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for n in node.names:
                aliases[n.asname or n.name] = f"{node.module}.{n.name}"
    targets = set(decorators)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            parts = _dotted(decorator.func if isinstance(decorator, ast.Call) else decorator)
            if not parts:
                continue
            raw = ".".join(parts)
            resolved = ".".join([aliases.get(parts[0], parts[0]), *parts[1:]])
            if raw in targets or resolved in targets:
                names.add(node.name)
    return names


def agent_kernel_names(workdir, backend=None) -> list[str]:
    """Authored kernel names for a backend, in stable name order."""
    root = Path(workdir)
    spec = get_spec(backend or detect_backend(root))
    paths = sorted(root.glob(spec.kernel_glob))
    if spec.decorators:
        names = set()
        for path in paths:
            names |= decorated_functions(path.read_text(), spec.decorators)
    else:
        names = _cuda_globals(paths)
    return sorted(names)
