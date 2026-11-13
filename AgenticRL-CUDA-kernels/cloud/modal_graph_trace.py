"""Detached H200 Nsight Systems traces of CUDA-graph capture and replay (diagnostic).

    modal run --detach cloud/modal_graph_trace.py --tasks stage6 --case-set acceptance

Runs scripts/graph_trace.py on one GPU and writes traces, worker records and
SVG timelines to the agenticrl-cuda-artifacts volume under /<run_id>/.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

from cloud.modal_job import wait_for_call
from cloud.modal_runtime import GPU, REMOTE, image, volume

app = modal.App("agenticrl-graph-trace", image=image)


@app.function(gpu=GPU, timeout=7200, volumes={"/artifacts": volume})
def trace(run_id: str, tasks: str, backends: str, case_set: str, replays: int) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a plain directory name")
    output = Path("/artifacts") / run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_AGENT_DEVICE": "cuda", "PYTHONDONTWRITEBYTECODE": "1"}
    request = {"run_id": run_id, "tasks": tasks, "backends": backends, "case_set": case_set,
               "replays": replays, "gpu_requested": GPU}
    command = ["python", "-m", "scripts.graph_trace", "--tasks", tasks, "--backends", backends,
               "--case-set", case_set, "--replays", str(replays), "--output-dir", str(output / "traces")]
    output.mkdir(exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    for name, probe in (("runtime", ["python", "-m", "utils.runtime", "--json", "--require", backends]),
                        ("nvidia-smi", ["nvidia-smi"])):
        proc = subprocess.run(probe, cwd=REMOTE, env=env, capture_output=True, text=True, timeout=300)
        (output / f"{name}.stdout.log").write_text(proc.stdout)
        (output / f"{name}.stderr.log").write_text(proc.stderr)
    proc = subprocess.run(command, cwd=REMOTE, env=env, capture_output=True, text=True, timeout=7000)
    (output / "trace.stdout.log").write_text(proc.stdout)
    (output / "trace.stderr.log").write_text(proc.stderr)
    summary = {"run_id": run_id, "command": command, "returncode": proc.returncode,
               "tail": proc.stdout.strip().splitlines()[-1:] if proc.stdout.strip() else []}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(tasks: str = "stage6", backends: str = "cuda,triton", case_set: str = "acceptance",
         replays: int = 5, run_id: str = ""):
    run_id = run_id or "graph-trace-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = trace.spawn(run_id, tasks, backends, case_set, replays)
    print(json.dumps({"run_id": run_id, "function_call_id": call.object_id,
                      "volume": "agenticrl-cuda-artifacts", "remote": f"/artifacts/{run_id}"}), flush=True)
    print(json.dumps(wait_for_call(call), allow_nan=False), flush=True)
