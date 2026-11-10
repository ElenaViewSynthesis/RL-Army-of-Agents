"""Freeze an allowlisted public source tree before a Modal task-suite upload."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from utils.task_suite import ROOT, select_tasks


PACKAGES = ("utils", "scripts", "cloud", "tracking", "kernels", "tests", "Architecture-decision", "legal")
SUFFIXES = frozenset((".py", ".cpp", ".cu", ".h", ".md", ".txt", ".sh"))
EXCLUDED = frozenset(("__pycache__", ".cache", ".artifacts", "build", ".git", ".venv"))
ROOT_FILES = (
    "AGENTS.md", "README.md", "VALIDATION.md", "KNOWN_ISSUES.md", "Benchmark_diagnostics_task.md",
    "EAGLE4_STAGE5.md", "MULTI_LORA_STAGE6.md", "MULTI_LORA_kernels_task.md", "SKILL.md", "SKILL_CUDA.md", "SKILL_TRITON.md", "SKILL_TILELANG.md", "SKILL_CUTILE.md", "SKILL_CUTEDSL.md",
    "model.py", "model_new.py", "model_new_triton.py", "model_new_tilelang.py",
    "model_new_cutile.py", "model_new_cutedsl.py", "binding.cpp", "binding_registry.h",
    "requirements.txt", "requirements-gpu.txt", "requirements-tilelang.txt",
    "requirements-cutedsl.txt", "requirements-cutile.txt",
)


def freeze(destination, task_names):
    destination = Path(destination).resolve()
    names = select_tasks(task_names)
    if not names:
        raise ValueError("Source freeze needs a nonempty task selection")
    paths = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    for folder in (*PACKAGES, *(f"tasks/{name}" for name in names)):
        for path in (ROOT / folder).rglob("*"):
            if (path.is_file() and (path.suffix in SUFFIXES or path.name in ("LICENSE", "NOTICE")
                                             or (folder == "legal" and path.name in ("source.json", "sources.json", "punica-LICENSE", "slora-LICENSE", "vllm-LICENSE")))
                    and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts)):
                paths.append(path)
    paths.append(ROOT / "artifacts" / "upstream.json")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in sorted(set(paths)):
        if source.is_symlink() or not source.is_relative_to(ROOT):
            raise ValueError(f"Refusing source outside the public source tree: {source}")
        relative = source.relative_to(ROOT)
        data = source.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    for relative, digest in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during freeze: {relative}; snapshot is not accepted")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    manifest = {"schema": "task-suite-source-v1", "tasks": names,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_revision": git.stdout.strip(), "files": hashes,
                "scope": "public source only; no .env, credentials, caches, binaries or old evidence"}
    (destination / "task-source-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for relative in (*hashes, "task-source-manifest.json"):
        (destination / relative).chmod(0o444)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="bootstrap,stage1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.output, args.tasks.split(","))
    print(json.dumps({"source": str(args.output.resolve()), "tasks": manifest["tasks"],
                      "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
