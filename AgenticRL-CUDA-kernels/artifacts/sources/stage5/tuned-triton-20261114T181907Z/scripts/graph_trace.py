"""Trace CUDA-graph capture and replay of honest task solutions with Nsight Systems.

For each task × backend: scaffold the honest fixture, compile it with the normal
compile stage, run utils/graph_trace_worker under
`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`, export SQLite and
render timelines with scripts/render_nsys_timeline.py. Diagnostic only: no
reward, verification gate or suite row changes. GPU host required.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from utils.backend import BACKENDS
from utils.rollout_env import _stage, bootstrap_script
from utils.task_catalog import get_task, select_cases
from utils.task_suite import scaffold, select_tasks

ROOT = Path(__file__).resolve().parent.parent
TRACED_SOURCES = ("utils/graph_trace_worker.py", "scripts/graph_trace.py", "scripts/render_nsys_timeline.py",
                  "utils/benchmark.py", "utils/verification.py", "utils/lora_verification.py")


def nsys_command(binary, output, script):
    return [binary, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none",
            "--cpuctxsw=none", "--export=sqlite", "--force-overwrite=true",
            "--output", str(output), sys.executable, "-c", script]


def case_env(task_name, case_id, workdir):
    spec = get_task(task_name)
    env = {**os.environ, "CUDA_AGENT_TASK": task_name, "CUDA_AGENT_TASK_CASE": case_id,
           "CUDA_AGENT_DEVICE": "cuda", "TRITON_INTERPRET": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "WANDB_MODE": "disabled", "WEAVE_DISABLED": "true",
           "CUDA_AGENT_RELATIVE_ERROR_BOUNDS":
               "1" if spec.stage == 6 and task_name != "lora_segment_metadata" else "0"}
    for backend in BACKENDS.values():
        for var, subdir in backend.cache_env:
            env[var] = str(workdir / ".cache" / subdir)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(workdir / ".cache" / "inductor")
    return env


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def trace_one(task_name, case_id, backend, out, *, replays, timeout_s, nsys="nsys"):
    out.mkdir(parents=True, exist_ok=True)
    work = scaffold(task_name, backend, "honest", out / "work")
    (work / ".cache").mkdir(exist_ok=True)
    env = case_env(task_name, case_id, work)
    record = {"task": task_name, "case": case_id, "backend": backend, "status": "failed",
              "candidate_sha256": _sha(work / "model_new.py")}
    deadline = time.monotonic() + timeout_s
    code, stdout, stderr, timed_out = _stage("utils.compile", ["--backend", backend, "--seed", "0"], work, env, deadline)
    (out / "compile.stdout.log").write_text(stdout)
    (out / "compile.stderr.log").write_text(stderr)
    if timed_out or code != 0:
        record["reason"] = "compile timeout" if timed_out else f"compile exited {code}"
        return record
    worker_json = out / "worker.json"
    script = bootstrap_script("utils.graph_trace_worker",
                              ["--output", str(worker_json), "--replays", str(replays)], work)
    argv = nsys_command(nsys, out / "trace", script)
    record["nsys_command"] = argv[:argv.index(sys.executable)] + ["python", "-c", "<bootstrap utils.graph_trace_worker>"]
    try:
        proc = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True,
                              timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        record["reason"] = "nsys timeout"
        return record
    (out / "nsys.stdout.log").write_text(proc.stdout)
    (out / "nsys.stderr.log").write_text(proc.stderr)
    record["nsys_returncode"] = proc.returncode
    if proc.returncode != 0 or not (out / "trace.sqlite").is_file() or not worker_json.is_file():
        record["reason"] = f"nsys exited {proc.returncode}" if proc.returncode else "missing trace or worker output"
        return record
    from scripts.render_nsys_timeline import render
    summary, files = render(out / "trace.sqlite", out, f"{task_name} · {backend} · {case_id}")
    worker = json.loads(worker_json.read_text())
    record.update(status="traced", worker=worker, svg=files,
                  phase_counts={impl: len(rows) for impl, rows in summary.items()})
    shutil.rmtree(work, ignore_errors=True)  # scaffold copy duplicates committed task files
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="stage6")
    parser.add_argument("--backends", default="cuda,triton")
    parser.add_argument("--case-set", default="acceptance")
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=1200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit("nsys not found on PATH")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index = {"schema": "graph-trace-run-v1", "case_set": args.case_set, "replays": args.replays,
             "nsys_version": subprocess.run([nsys, "--version"], capture_output=True, text=True).stdout.strip(),
             "sources_sha256": {name: _sha(ROOT / name) for name in TRACED_SOURCES}, "traces": []}
    for task_name in select_tasks(args.tasks.split(",")):
        case_id = select_cases(task_name, args.case_set)[0].id
        for backend in [b for b in args.backends.split(",") if b in get_task(task_name).backends]:
            out = args.output_dir / task_name / backend
            try:
                record = trace_one(task_name, case_id, backend, out, replays=args.replays,
                                   timeout_s=args.timeout_s, nsys=nsys)
            except Exception as exc:  # one failed trace never stops the rest
                record = {"task": task_name, "case": case_id, "backend": backend, "status": "failed",
                          "reason": f"{type(exc).__name__}: {exc}"[:512]}
            index["traces"].append(record)
            print(json.dumps({k: record.get(k) for k in ("task", "backend", "status", "reason")}), flush=True)
            (args.output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    traced = sum(r["status"] == "traced" for r in index["traces"])
    print(json.dumps({"traced": traced, "total": len(index["traces"])}))


if __name__ == "__main__":
    main()
