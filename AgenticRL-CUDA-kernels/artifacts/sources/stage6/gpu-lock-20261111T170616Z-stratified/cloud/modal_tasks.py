"""Detached, frozen-source H200 workload validation using durable spawn/poll."""

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

from cloud.modal_job import wait_for_call
from cloud.modal_runtime import GPU, REMOTE, image, volume
from utils.task_suite import select_tasks


app = modal.App("agenticrl-progressive-kernel-tasks", image=image)


def source_check(root, manifest):
    return [name for name, digest in manifest["files"].items()
            if not (root / name).is_file()
            or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]


def run_logged(command, root, output, timeout_s, environment):
    stdout_path, stderr_path = output / "suite.stdout.log", output / "suite.stderr.log"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    volume.commit()
                    progress = output / "suite" / "progress.jsonl"
                    try:
                        lines = progress.read_text().splitlines()
                        if lines:
                            last = json.loads(lines[-1])
                            print(json.dumps({"completed": len(lines), "last_task": last["suite/task"],
                                              "last_status": last["suite/status"]}), flush=True)
                    except (OSError, ValueError, KeyError):
                        pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                timed_out = True
    return {"command": shlex.join(command), "returncode": process.returncode,
            "timed_out": timed_out, "wall_s": time.monotonic() - started}


@app.function(gpu=GPU, timeout=21600, volumes={"/artifacts": volume})
def evaluate(run_id: str, tasks: str, case_set: str, backends: str, fixtures: str,
             timeout_s: int, benchmark: bool, cold_start: bool, cuda_graphs: bool = False) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a plain directory name")
    root, output = Path(REMOTE), Path("/artifacts") / run_id
    manifest = json.loads((root / "task-source-manifest.json").read_text())
    names = select_tasks(tasks.split(","))
    if set(names) - set(manifest["tasks"]):
        raise ValueError("Requested task was not included in the frozen source")
    if source_check(root, manifest):
        raise ValueError("Uploaded source does not match the frozen manifest")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for relative in (*manifest["files"], "task-source-manifest.json"):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    environment = {**os.environ, "CUDA_AGENT_DEVICE": "cuda", "WANDB_MODE": "offline",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    request = {"run_id": run_id, "tasks": names, "case_set": case_set, "backends": backends,
               "fixtures": fixtures, "timeout_s": timeout_s, "benchmark": benchmark,
               "cold_start": cold_start, "cuda_graphs": cuda_graphs, "gpu_requested": GPU}
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    for name, command in (
            ("runtime", ["python", "-m", "utils.runtime", "--json", "--require", backends or "cuda,triton"]),
            ("nvidia-smi", ["nvidia-smi"])):
        probe = subprocess.run(command, cwd=root, env=environment,
                               capture_output=True, text=True, timeout=300)
        (output / f"{name}.stdout.log").write_text(probe.stdout)
        (output / f"{name}.stderr.log").write_text(probe.stderr)
    command = ["python", "-m", "scripts.validate", "--suite", ",".join(names),
               "--device", "cuda", "--purpose", "eval", "--case-set", case_set,
               "--fixtures", fixtures, "--timeout-s", str(timeout_s),
               "--suite-out", str(output / "suite")]
    if backends:
        command += ["--backends", backends]
    if benchmark:
        command.append("--benchmark")
    if cold_start:
        command.append("--cold-start")
    if cuda_graphs:
        command.append("--cuda-graphs")
    execution = run_logged(command, root, output, 20700, environment)
    mismatches = source_check(root, manifest)
    try:
        suite = json.loads((output / "suite" / "suite.json").read_text())
    except (OSError, ValueError):
        suite = {}
    summary = {"run_id": run_id, "execution": execution, "counts": suite.get("counts"),
               "status": suite.get("status", "incomplete"), "source_mismatches": mismatches,
               "scope": "selected workload cases only; no stock AXPBY acceptance or timing-cause claim"}
    if execution["timed_out"] or execution["returncode"] not in (0, 1) or mismatches:
        summary["status"] = "incomplete" if not mismatches else "fail"
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(tasks: str = "bootstrap,stage1", case_set: str = "smoke", backends: str = "cuda,triton",
         fixtures: str = "honest,hack,wrong", timeout_s: int = 600, benchmark: bool = True,
         cold_start: bool = True, run_id: str = "", cuda_graphs: bool = False):
    if cuda_graphs and not benchmark:
        raise ValueError("cuda_graphs requires benchmark")
    if cold_start and not benchmark:
        raise ValueError("cold_start requires benchmark")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    call = evaluate.spawn(run_id, tasks, case_set, backends, fixtures, timeout_s, benchmark, cold_start, cuda_graphs)
    print(json.dumps({"run_id": run_id, "function_call_id": call.object_id,
                      "volume": "agenticrl-cuda-artifacts", "remote": f"/artifacts/{run_id}"}), flush=True)
    print(json.dumps(wait_for_call(call), allow_nan=False), flush=True)
