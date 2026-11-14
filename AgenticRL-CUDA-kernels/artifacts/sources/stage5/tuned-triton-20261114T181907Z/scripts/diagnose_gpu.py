import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from tests.support import ROOT, make_task, scratch
from utils.backend import BACKENDS
from utils.diagnose import CHECKS, TOOLS, run_diagnostic


def plan(backends, tools, checks):
    rows = [("sanitizer", backend, check, False)
            for backend in backends for check in checks] if "sanitizer" in tools else []
    if "sanitizer" in tools and "memcheck" in checks:
        rows.append(("sanitizer", "triton", "memcheck", True))
    rows += [(tool, backend, None, False) for tool in ("nsys", "ncu") if tool in tools
             for backend in backends]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--tools", default=",".join(TOOLS))
    parser.add_argument("--checks", default=",".join(CHECKS))
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    backends, tools, checks = (value.split(",") for value in (args.backends, args.tools, args.checks))
    for values, known in ((backends, BACKENDS), (tools, TOOLS), (checks, CHECKS)):
        if not values or len(set(values)) != len(values) or any(value not in known for value in values):
            parser.error("Known, distinct backends/tools/checks required")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be positive and finite")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates = output / "candidates"
    candidates.mkdir()
    jobs = plan(backends, tools, checks)
    manifest = {"schema": "gpu-diagnostic-matrix-v1", "diagnostic_only": True,
                "run_id": args.run_id, "argv": sys.argv, "planned": jobs,
                "scope": backends, "tools": tools, "checks": checks,
                "timeout_s": args.timeout_s, "seed": args.seed}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with scratch("gpu-diagnostics-") as tmp:
        needed = list(dict.fromkeys([backend for _, backend, _, _ in jobs]))
        tasks = {backend: make_task(tmp, backend) for backend in needed}
        for backend, task in tasks.items():
            candidate = (task / "model_new.py").read_bytes()
            (candidates / f"{backend}.py").write_bytes(candidate)
        fault = (ROOT / "utils/diagnostic_fault.py").read_bytes()
        (candidates / "invalid-access.py").write_bytes(fault)
        for index, (tool, backend, check, negative) in enumerate(jobs):
            name = f"{index:02d}-{tool}-{backend}-{check or 'capture'}" + ("-invalid-access" if negative else "")
            relative = Path("cases") / name / "report.json"
            result = run_diagnostic(tool, tasks[backend], output / relative.parent,
                                    backend=backend, check=check or "memcheck",
                                    timeout_s=args.timeout_s, seed=args.seed,
                                    negative_control=negative)
            candidate = fault if negative else (tasks[backend] / "model_new.py").read_bytes()
            row = {"case": name, "tool": tool, "backend": backend, "check": check,
                   "negative_control": negative, "report": str(relative),
                   "candidate_sha256": hashlib.sha256(candidate).hexdigest(), "result": result}
            rows.append(row)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    (output / "records.json").write_text(json.dumps(
        {"diagnostic_only": True, "run_id": args.run_id, "rows": rows},
        indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
