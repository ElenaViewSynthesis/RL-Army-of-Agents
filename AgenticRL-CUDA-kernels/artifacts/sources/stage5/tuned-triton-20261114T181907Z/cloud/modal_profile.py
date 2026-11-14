"""Profile honest task solutions on a Modal Blackwell GPU with Nsight Systems / Nsight Compute.

Diagnostic only: it runs scripts.profile_tasks on the GPU host and saves everything to the
agenticrl-cuda-artifacts volume under /<run_id>/. Nothing here feeds a reward, and a
failing row is recorded while the rest of the matrix continues.

MODAL_GPU selects the GPU and defaults to B200. The image is CUDA 13.5 devel on Ubuntu 24.04
(it ships nsys) and builds CUDA C++ extensions for sm_100 (TORCH_CUDA_ARCH_LIST=10.0). The
H200 task image in cloud/modal_runtime.py is unchanged. This repo's CUDA C++ fixtures are
validated only on sm_90, so Triton is the default backend here.

    MODAL_GPU=B200 modal run --detach cloud/modal_profile.py \
        --tasks relu_squared,rmsnorm --backends triton --case-set smoke --tools nsys

    modal volume get agenticrl-cuda-artifacts /<run_id>/ artifacts/runs/<run_id>
    python -m cloud.modal_job --call-id fc-...       # re-attach if the local command died

A paid GPU run: launch it only with the user's approval. scripts/profile_b200.sh wraps
launch, fetch and query; `python -m scripts.profile_tasks --dry-run ...` prints the plan free.
"""

import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

from cloud.modal_job import wait_for_call

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/AgenticRL-CUDA-kernels"
GPU = os.environ.get("MODAL_GPU", "B200")
ARCH = os.environ.get("MODAL_PROFILE_ARCH", "10.0")  # set 9.0 with MODAL_GPU=H200

image = (
    modal.Image.from_registry("nvidia/cuda:13.5.0-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("ninja-build", "git", "libdw1")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": ARCH, "PYTHONPATH": REMOTE})
    .add_local_dir(
        str(ROOT), remote_path=REMOTE,
        ignore=[".venv", ".cache", "build", "__pycache__", "artifacts", ".env", ".env.*", "*.so", "*.pyc"])
)

volume = modal.Volume.from_name("agenticrl-cuda-artifacts", create_if_missing=True)
app = modal.App("agenticrl-task-profile", image=image)
PROBES = (("nvidia-smi", ["nvidia-smi"]),
          ("gpu-query", ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,uuid", "--format=csv,noheader"]),
          ("nsys-version", ["nsys", "--version"]), ("ncu-version", ["ncu", "--version"]),
          ("nvcc-version", ["nvcc", "--version"]))


def probe(out, name, argv):
    """Record one host fact; a missing tool is logged, not raised."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        code, stdout, stderr = None, "", f"{type(exc).__name__}: {exc}"
    (out / f"{name}.stdout.log").write_text(stdout)
    (out / f"{name}.stderr.log").write_text(stderr)
    return {"returncode": code, "first_line": stdout.strip().splitlines()[0] if stdout.strip() else None}


@app.function(gpu=GPU, timeout=21600, volumes={"/artifacts": volume})
def profile(run_id: str, tasks: str, backends: str = "triton", case_set: str = "smoke", tools: str = "nsys",
            nsys_profile: str = "trace", ncu_preset: str = "basic", timeout_s: int = 900) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a plain directory name")
    out = Path("/artifacts") / run_id
    out.mkdir(parents=True, exist_ok=False)
    summary = {"run_id": run_id, "diagnostic_only": True, "gpu_requested": GPU, "torch_cuda_arch_list": ARCH,
               "tasks": tasks, "backends": backends, "case_set": case_set, "tools": tools,
               "nsys_profile": nsys_profile, "ncu_preset": ncu_preset, "timeout_s": timeout_s}
    (out / "request.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["host"] = {name: probe(out, name, argv) for name, argv in PROBES}
    volume.commit()

    command = ["python", "-m", "scripts.profile_tasks", "--tasks", tasks, "--backends", backends,
               "--case-set", case_set, "--tools", tools, "--nsys-profile", nsys_profile,
               "--ncu-preset", ncu_preset, "--timeout-s", str(timeout_s), "--run-id", run_id,
               "--output-dir", str(out / "profile")]
    env = {**os.environ, "CUDA_AGENT_DEVICE": "cuda", "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled"}
    started, timed_out = time.monotonic(), False
    progress = out / "profile" / "progress.jsonl"
    with (out / "profile.stdout.log").open("wb") as stdout, (out / "profile.stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=REMOTE, env=env, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            if time.monotonic() - started > 20700:
                timed_out = True
                process.kill()
                break
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                volume.commit()  # partial captures survive a killed container
                try:
                    lines = progress.read_text().splitlines()
                    last = json.loads(lines[-1])
                    print(json.dumps({"run_id": run_id, "completed": len(lines), "task": last.get("task"),
                                      "backend": last.get("backend"), "tool": last.get("tool"),
                                      "status": (last.get("result") or {}).get("diag/status")}), flush=True)
                except (OSError, ValueError, IndexError):
                    pass
        process.wait()
    summary["execution"] = {"command": shlex.join(command), "returncode": process.returncode,
                            "timed_out": timed_out, "wall_s": round(time.monotonic() - started, 3)}
    try:
        records = json.loads((out / "profile" / "records.json").read_text())
        summary["rows"], summary["statuses"] = records["rows"], records["statuses"]
    except (OSError, ValueError, KeyError):
        summary["rows"], summary["statuses"] = None, None  # the driver did not finish; see profile.stderr.log
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(tasks: str, backends: str = "triton", case_set: str = "smoke", tools: str = "nsys",
         nsys_profile: str = "trace", ncu_preset: str = "basic", timeout_s: int = 900, run_id: str = ""):
    run_id = run_id or "profile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    call = profile.spawn(run_id, tasks, backends, case_set, tools, nsys_profile, ncu_preset, timeout_s)
    print(json.dumps({"run_id": run_id, "function_call_id": call.object_id, "gpu": GPU,
                      "volume": "agenticrl-cuda-artifacts", "remote": f"/artifacts/{run_id}"}), flush=True)
    print(json.dumps(wait_for_call(call), allow_nan=False, default=str), flush=True)
