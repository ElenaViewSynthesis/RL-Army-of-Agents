"""Profile honest task solutions on the GPU this process can see: diagnostic only.

For every selected (task, case, backend) it scaffolds the honest fixture and runs
`utils.diagnose` with Nsight Systems and/or Nsight Compute, then answers the usual
questions about each Nsight Systems capture with scripts.query_nsys (summary,
kernels, nvtx, memcpy) and saves the JSON next to it. Nothing here feeds a reward.

    python -m scripts.profile_tasks --tasks relu_squared,rmsnorm --backends triton \
        --case-set smoke --tools nsys --output-dir <out>
    python -m scripts.profile_tasks --tasks stage7 --dry-run        # print the plan; no GPU, no files

It is GPU-agnostic; cloud/modal_profile.py runs it on a Modal B200 and
scripts/profile_b200.sh wraps launch, fetch and query. Output layout:

    <out>/manifest.json              the request and the planned rows
    <out>/progress.jsonl             one line per finished row
    <out>/records.json               every row, written at the end
    <out>/cases/<nn>-<task>-<case>-<backend>-<tool>/   report.json, capture.*, logs, query-*.json

A backend with no honest fixture for a task is recorded as skipped with a reason,
never silently dropped. The last stdout line is the JSON summary.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from utils.backend import BACKENDS
from utils.diagnose import NCU_PRESETS, parse_profiles
from utils.task_catalog import get_task, select_cases

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ("nsys", "ncu")
QUERIES = ("summary", "kernels", "nvtx", "memcpy")


def fixture(task_name, backend):
    return ROOT / "tasks" / task_name / f"model_new_{backend}.py"


def plan(tasks, backends, case_set, tools):
    """Rows in run order. Each names a task, case, backend and tool, plus a skip reason or None."""
    rows = []
    for task_name in tasks:
        task = get_task(task_name)
        for case in select_cases(task_name, case_set):
            for backend in backends:
                if task.contract_only:
                    skip = "contract_only_task"
                elif not fixture(task_name, backend).is_file():
                    skip = "no_honest_fixture_for_backend"
                else:
                    skip = None
                for tool in tools:
                    rows.append({"task": task_name, "case": case.id, "backend": backend, "tool": tool, "skip": skip})
    return rows


def case_name(index, row):
    return f"{index:03d}-{row['task']}-{row['case']}-{row['backend']}-{row['tool']}"


def query_capture(sqlite, out, top):
    """Save each query_nsys report beside the capture; return the kernel totals for the summary line."""
    from scripts import query_nsys

    reports = {"summary": query_nsys.event_summary(sqlite), "kernels": query_nsys.kernel_stats(sqlite, top=top),
               "nvtx": query_nsys.nvtx_stats(sqlite), "memcpy": query_nsys.memcpy_stats(sqlite)}
    for name in QUERIES:
        (out / f"query-{name}.json").write_text(json.dumps(reports[name], indent=2, allow_nan=False) + "\n")
    kernels = reports["kernels"]
    return {"kernel_launches": kernels["launches"], "kernel_total_ns": kernels["total_ns"],
            "top_kernel": kernels["kernels"][0]["name"] if kernels["kernels"] else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True, help="comma-separated task names or groups (stage1..stage7, bootstrap, all)")
    parser.add_argument("--backends", default="triton", help=f"comma-separated subset of {','.join(BACKENDS)}")
    parser.add_argument("--case-set", default="smoke", help="smoke, acceptance, stratified, all, or one case ID")
    parser.add_argument("--tools", default="nsys", help=f"comma-separated subset of {','.join(TOOLS)}")
    parser.add_argument("--nsys-profile", default="trace", help="utils.diagnose nsys --profile value")
    parser.add_argument("--ncu-preset", default="basic", choices=tuple(NCU_PRESETS))
    parser.add_argument("--timeout-s", type=float, default=900, help="per row and tool")
    parser.add_argument("--top", type=int, default=20, help="kernels kept in query-kernels.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the plan as JSON and exit; needs no GPU")
    args = parser.parse_args(argv)

    from utils.task_suite import select_tasks

    backends, tools = args.backends.split(","), args.tools.split(",")
    for values, known in ((backends, BACKENDS), (tools, TOOLS)):
        if not values or len(set(values)) != len(values) or any(value not in known for value in values):
            parser.error("known, distinct --backends and --tools required")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0 or args.top <= 0:
        parser.error("--timeout-s and --top must be positive")
    try:
        parse_profiles(args.nsys_profile)
        tasks = select_tasks(args.tasks.split(","))
        rows = plan(tasks, backends, args.case_set, tools)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    request = {"schema": "task-profile-matrix-v1", "diagnostic_only": True, "run_id": args.run_id, "tasks": tasks,
               "backends": backends, "case_set": args.case_set, "tools": tools, "nsys_profile": args.nsys_profile,
               "ncu_preset": args.ncu_preset, "timeout_s": args.timeout_s, "seed": args.seed}
    if args.dry_run:
        print(json.dumps({**request, "planned": rows, "to_run": sum(row["skip"] is None for row in rows),
                          "skipped": sum(row["skip"] is not None for row in rows)}))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run")

    from tests.support import scratch
    from utils.diagnose import run_diagnostic
    from utils.task_suite import case_environment, scaffold

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps({**request, "argv": sys.argv, "planned": rows}, indent=2) + "\n")
    records, workdirs = [], {}
    with scratch("profile-tasks-") as tmp:
        for index, row in enumerate(rows):
            name = case_name(index, row)
            record = {"case_dir": name, **row}
            if row["skip"] is None:
                key = (row["task"], row["backend"])
                if key not in workdirs:
                    workdirs[key] = scaffold(row["task"], row["backend"], "honest",
                                             Path(tmp) / f"{row['task']}-{row['backend']}")
                workdir, out = Path(workdirs[key]), output / "cases" / name
                record["candidate_sha256"] = hashlib.sha256((workdir / "model_new.py").read_bytes()).hexdigest()
                try:
                    with case_environment(row["task"], row["case"], "cuda"):
                        record["result"] = run_diagnostic(
                            row["tool"], workdir, out, backend=row["backend"], timeout_s=args.timeout_s,
                            seed=args.seed, profile=args.nsys_profile if row["tool"] == "nsys" else "trace",
                            preset=args.ncu_preset if row["tool"] == "ncu" else "basic")
                    record["report"] = f"cases/{name}/report.json"
                except Exception as exc:  # one bad row is recorded; the matrix continues
                    record["result"] = None
                    record["driver_error"] = f"{type(exc).__name__}: {exc}"
                sqlite = out / "capture.sqlite"
                if row["tool"] == "nsys" and sqlite.is_file():
                    try:
                        record["query"] = query_capture(sqlite, out, args.top)
                    except Exception as exc:  # a query failure must not lose the capture's record
                        record["query"] = None
                        record["query_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
    statuses = {}
    for record in records:
        status = ("skipped" if record["skip"] else "driver_error" if record["result"] is None
                  else record["result"].get("diag/status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    summary = {"diagnostic_only": True, "run_id": args.run_id, "rows": len(records), "statuses": statuses}
    (output / "records.json").write_text(json.dumps({**summary, "request": request, "records": records},
                                                    indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
