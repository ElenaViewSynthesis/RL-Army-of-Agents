"""Torch-free assembly of opt-in diagnostic profiling artifacts.

The diagnostic pass (utils.profile_artifacts) writes report.json,
kernels.json and, on a GPU, trace.json under <workdir>/.artifacts/<run_id>/.
Everything here is diagnostics only: the reward never reads these files,
and a candidate can modify its own workdir, so they are not trusted inputs.
Kernel classes reuse utils.provenance unchanged; they are name-based
heuristics, not proof of authorship.
"""

import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from utils.provenance import PROVENANCE_RULESET, classify_kernel

SCHEMA_VERSION = 1
SHARES = ("agent", "library", "torch", "memory")
TOOLS = ("nvidia-smi", "nvcc", "ninja", "nsys", "ncu", "compute-sanitizer")
HARNESS_ROOT = Path(__file__).resolve().parent.parent
NOTES = ("Diagnostic pass only: profiler instrumentation perturbs timings, which never feed the reward.",
         "Kernel classes are name-based heuristics, not proof of authorship; unknown names count as torch.",
         "Workdir files are writable by the candidate; this report is not a trusted reward input.")


def kernel_table(events, agent_names, backend, iters):
    """Per-kernel rows from (name, device_us) leaf CUDA events.

    calls/total_us cover the whole capture; per_iter_us divides by iters.
    Rows are sorted by total device time, largest first.
    """
    if iters <= 0:
        raise ValueError("iters must be positive")
    totals = defaultdict(float)
    calls = defaultdict(int)
    for name, duration in events:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Profiler duration must be finite and non-negative")
        totals[name] += duration
        calls[name] += 1
    rows = [{"name": name, "calls": calls[name], "total_us": totals[name],
             "mean_us": totals[name] / calls[name], "per_iter_us": totals[name] / iters,
             "class": classify_kernel(name, agent_names, backend)}
            for name in totals]
    return sorted(rows, key=lambda row: (-row["total_us"], row["name"]))


def provenance_summary(rows):
    """Totals (us over the capture) and shares of total device time per class.

    `provenance` matches utils.provenance: agent + library over non-memory time.
    Shares are None when nothing was captured, never a fabricated 0.
    """
    totals = {k: 0.0 for k in SHARES}
    for row in rows:
        totals[row["class"]] += row["total_us"]
    total = sum(totals.values())
    non_memory = total - totals["memory"]
    return {"ruleset": PROVENANCE_RULESET,
            "totals_us": totals,
            "shares": {k: (v / total if total > 0 else None) for k, v in totals.items()},
            "provenance": ((totals["agent"] + totals["library"]) / non_memory
                           if non_memory > 0 else None)}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(root):
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def source_hashes(workdir, harness_root=HARNESS_ROOT):
    """SHA-256 of model.py, model_new.py, workdir kernels/** and the trusted utils/ harness.

    Keys are "task/<relpath>" and "harness/<relpath>"; a missing task file
    maps to None so its absence is visible.
    """
    workdir, harness_root = Path(workdir), Path(harness_root)
    hashes = {}
    for name in ("model.py", "model_new.py"):
        path = workdir / name
        hashes[f"task/{name}"] = _sha256(path) if path.is_file() else None
    for path in _tree(workdir / "kernels"):
        hashes["task/" + path.relative_to(workdir).as_posix()] = _sha256(path)
    for path in _tree(harness_root / "utils"):
        hashes["harness/" + path.relative_to(harness_root).as_posix()] = _sha256(path)
    return hashes


def tool_capabilities(which=shutil.which, cupti_events=None, cupti_reason=None):
    """Which external tools are on PATH, and whether CUPTI produced CUDA events.

    cupti_events is the CUDA event count from the diagnostic capture, or None
    when no capture ran (cupti_reason then says why).
    """
    report = {}
    for tool in TOOLS:
        path = which(tool)
        report[tool] = ({"available": True, "path": path} if path else
                        {"available": False, "reason": "not found on PATH"})
    if cupti_events is None:
        report["cupti"] = {"available": False, "reason": cupti_reason or "no CUDA capture ran"}
    elif cupti_events > 0:
        report["cupti"] = {"available": True, "cuda_events": cupti_events}
    else:
        report["cupti"] = {"available": False, "cuda_events": 0,
                           "reason": "profiler produced no CUDA events"}
    return report


def build_report(*, run_id, task_id, backend, started_at, finished_at, workdir,
                 inputs=None, inputs_reason=None, gpu=None, gpu_reason=None,
                 software=None, capture=None, rows=None, benchmark=None,
                 capabilities=None, trace=None, trace_reason=None):
    """Assemble report.json; raises ValueError if it is not strict JSON."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "task_id": task_id,
        "backend": backend,
        "notes": list(NOTES),
        "inputs": inputs,
        "inputs_reason": None if inputs is not None else (inputs_reason or "not recorded"),
        "gpu": gpu,
        "gpu_reason": None if gpu is not None else (gpu_reason or "not recorded"),
        "software": {"python": sys.version.split()[0], **(software or {})},
        "source_sha256": source_hashes(workdir),
        "capture": capture,
        "provenance": provenance_summary(rows) if rows else None,
        "benchmark": benchmark if benchmark is not None else {"enabled": False},
        "capabilities": capabilities if capabilities is not None else tool_capabilities(),
        "trace": trace,
        "trace_reason": None if trace is not None else (trace_reason or "not captured"),
    }
    json.dumps(report, allow_nan=False)
    return report


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path
