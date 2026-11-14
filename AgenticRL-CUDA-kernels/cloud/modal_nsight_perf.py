"""Nsight Perf SDK for AI on a Modal Blackwell GPU, next to Nsight Systems: diagnostic only.

The SDK (https://github.com/NVIDIA/nsight-perf-sdk-for-ai, "Beta Preview") supports
Blackwell and Rubin only, needs Ubuntu 24.04 and CUDA 13.5+, and a 620+ driver or
a 580.95.05+ kernel driver with the CUDA 13.5 compatibility package. So this app
uses its own image (nvidia/cuda:13.5.0-devel-ubuntu24.04, which ships nsys) and
defaults to MODAL_GPU=B200; the H200 task image in cloud/modal_runtime.py is unchanged.

One run, for one honest Triton task case:
  1. record nvidia-smi, the driver, nsys and the SDK tools;
  2. run the workload alone (scripts.nsight_perf_workload);
  3. start the nvidia-nsight-perf service, falling back to CUDA/NVTX-only providers
     when the GPU PM / HW-trace providers do not come up;
  4. run the workload under a streaming consumer, then take a snapshot, then export
     each Trace Stream Database to Perfetto; when the driver is older than 620, a
     second pass loads the workload with the 13.5 compatibility driver;
  5. stop the service and capture the same case with `utils.diagnose nsys`
     (default trace profile plus the post-capture nsys stats/analyze reports).
Every step is logged to the agenticrl-cuda-artifacts volume under /<run_id>/; a
failing step is recorded and the run continues. Nothing here feeds a reward.

    MODAL_GPU=B200 modal run --detach cloud/modal_nsight_perf.py --run-id <id>
"""

import json
import os
import re
import shlex
import signal
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
SDK_COMMIT = "bd5d5901c6c9b72704245dd0f6e8e1a822bda527"  # main on 2026-11-14 (no tagged release)
SDK = "/opt/nsight-perf-sdk-for-ai"
BIN = f"{SDK}/build/bin"
COMPAT = "/usr/local/cuda/compat"

image = (
    modal.Image.from_registry("nvidia/cuda:13.5.0-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "cmake", "build-essential", "ninja-build", "libdw1", "curl", "ca-certificates")
    .run_commands(
        f"git clone https://github.com/NVIDIA/nsight-perf-sdk-for-ai.git {SDK}",
        f"cd {SDK} && git checkout {SDK_COMMIT} && git submodule update --init --recursive",
        f"cd {SDK} && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8",
        f"ls -la {BIN}",
    )
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0", "PYTHONPATH": REMOTE})
    .add_local_dir(
        str(ROOT), remote_path=REMOTE,
        ignore=[".venv", ".cache", "build", "__pycache__", "artifacts", ".env", ".env.*", "*.so", "*.pyc"])
)

volume = modal.Volume.from_name("agenticrl-cuda-artifacts", create_if_missing=True)
app = modal.App("agenticrl-nsight-perf-sdk", image=image)


class Steps:
    """Runs and logs each step; never raises on a failing command."""

    def __init__(self, out):
        self.out, self.records = out, []

    def run(self, name, argv, timeout_s, env=None, cwd=REMOTE):
        started, timed_out = time.monotonic(), False
        try:
            proc = subprocess.run(argv, cwd=cwd, env=env or os.environ.copy(), capture_output=True,
                                  text=True, timeout=timeout_s)
            code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out, code = True, None
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        except OSError as exc:
            code, stdout, stderr = None, "", f"{type(exc).__name__}: {exc}"
        (self.out / f"{name}.stdout.log").write_text(stdout)
        (self.out / f"{name}.stderr.log").write_text(stderr)
        record = {"step": name, "command": shlex.join(argv), "returncode": code, "timed_out": timed_out,
                  "wall_s": round(time.monotonic() - started, 3)}
        self.records.append(record)
        volume.commit()
        return record, stdout

    def start(self, name, argv, env=None, cwd=REMOTE):
        stdout = (self.out / f"{name}.stdout.log").open("wb")
        stderr = (self.out / f"{name}.stderr.log").open("wb")
        proc = subprocess.Popen(argv, cwd=cwd, env=env or os.environ.copy(), stdout=stdout, stderr=stderr,
                                start_new_session=True)
        return proc, (stdout, stderr), time.monotonic()

    def finish(self, name, argv, handle, timeout_s, stop=False):
        proc, files, started = handle
        timed_out = False
        if stop and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = not stop
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        for stream in files:
            stream.close()
        record = {"step": name, "command": shlex.join(argv), "returncode": proc.returncode,
                  "timed_out": timed_out, "stopped": stop, "wall_s": round(time.monotonic() - started, 3)}
        self.records.append(record)
        volume.commit()
        return record


def task_environment(task, case, compat=False):
    env = {**os.environ, "CUDA_AGENT_TASK": task, "CUDA_AGENT_TASK_CASE": case, "CUDA_AGENT_DEVICE": "cuda",
           "TRITON_INTERPRET": "0", "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
           # The workload's NVTX ranges must reach the tracer (as utils.diagnose sets for nsys).
           "NSYS_NVTX_PROFILER_REGISTER_ONLY": "0"}
    if compat:
        env["LD_LIBRARY_PATH"] = COMPAT + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def start_service(steps, attempt, reduced):
    env = os.environ.copy()
    if reduced:
        env.update(NSIGHT_PERF_GPU_PM_ENABLE="0", NSIGHT_PERF_GPU_HW_TRACE_ENABLE="0")
    argv = [f"{BIN}/nvidia-nsight-perf"]
    service = steps.start(f"service-{attempt}", argv, env=env)
    poll = [f"{BIN}/nvtrace_poll_until_service_ready", *(["--ignore", "gpu_pm,gpu_hwt"] if reduced else [])]
    ready, _ = steps.run(f"service-{attempt}-ready", ["timeout", "120", *poll], 130)
    return argv, service, ready["returncode"] == 0


def export_databases(steps, directory, label):
    exported = []
    for database in sorted(path for path in directory.iterdir() if path.is_dir()):
        record, _ = steps.run(f"export-{label}-{database.name}",
                              [f"{BIN}/nvtrace_perfetto_trace_exporter", "--tsf-database-dir", str(database)],
                              600, cwd=str(directory))
        exported.append(record)
    files = sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())
    sizes = {name: (directory / name).stat().st_size for name in files}
    return exported, sizes


@app.function(gpu=GPU, timeout=3600, volumes={"/artifacts": volume})
def trace(run_id: str, task: str = "relu_squared", case: str = "b4096_h14336_bfloat16",
          backend: str = "triton", iters: int = 20000, duration_ms: int = 30000) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a plain directory name")
    from utils.task_suite import scaffold

    out = Path("/artifacts") / run_id
    out.mkdir(parents=True, exist_ok=False)
    steps = Steps(out)
    summary = {"run_id": run_id, "diagnostic_only": True, "gpu_requested": GPU, "task": task, "case": case,
               "backend": backend, "sdk_commit": SDK_COMMIT, "iters": iters, "duration_ms": duration_ms}
    (out / "request.json").write_text(json.dumps(summary, indent=2) + "\n")

    steps.run("nvidia-smi", ["nvidia-smi"], 60)
    _, query = steps.run("gpu-query", ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,uuid",
                                       "--format=csv,noheader"], 60)
    steps.run("nsys-version", ["nsys", "--version"], 60)
    steps.run("nvcc-version", ["nvcc", "--version"], 60)
    steps.run("sdk-bin", ["ls", "-la", BIN], 60)
    steps.run("compat-libs", ["ls", "-la", COMPAT], 60)
    steps.run("uname", ["uname", "-a"], 60)
    driver = query.split(",")[1].strip() if query.count(",") >= 2 else None
    summary["gpu_query"] = query.strip()
    summary["driver_version"] = driver
    try:
        compat_needed = int(driver.split(".")[0]) < 620
    except (AttributeError, ValueError):
        compat_needed = None
    summary["compat_pass"] = bool(compat_needed)

    workdir = scaffold(task, backend, "honest", Path("/tmp") / f"{task}-{backend}")
    workload = ["python", "-m", "scripts.nsight_perf_workload", "--iters", str(iters)]
    env = task_environment(task, case)
    record, stdout = steps.run("workload-alone", workload, 900, env=env, cwd=str(workdir))
    summary["workload_alone"] = {**record, "result": stdout.strip().splitlines()[-1] if stdout.strip() else None}

    argv, service, ready = start_service(steps, "full", reduced=False)
    summary["service_mode"] = "full" if ready else None
    if not ready:
        steps.finish("service-full", argv, service, 30, stop=True)
        argv, service, ready = start_service(steps, "reduced", reduced=True)
        summary["service_mode"] = "reduced_no_gpu_pm_no_hwt" if ready else "not_ready"
    passes = {}
    if ready:
        for name, compat in (("native", False), *((("compat", True),) if compat_needed else ())):
            tsf = out / f"tsf-{name}"
            tsf.mkdir()
            consumer_argv = [f"{BIN}/nvtrace_streaming_consumer", "--output-dir", str(tsf),
                             "--duration-ms", str(duration_ms)]
            consumer = steps.start(f"consumer-{name}", consumer_argv)
            time.sleep(3)  # the consumer must be attached before the region of interest
            record, stdout = steps.run(f"workload-{name}", workload, 900, env=task_environment(task, case, compat),
                                       cwd=str(workdir))
            consumer_record = steps.finish(f"consumer-{name}", consumer_argv, consumer, duration_ms / 1000 + 120)
            snapshot = out / f"snapshot-{name}"
            snapshot.mkdir()
            snapshot_record, _ = steps.run(f"snapshot-{name}", [f"{BIN}/nvtrace_snapshot_consumer", "--output-dir",
                                                                 str(snapshot), "--duration-ms", "1000"], 300)
            exported, sizes = export_databases(steps, tsf, name)
            snapshot_exported, snapshot_sizes = export_databases(steps, snapshot, f"snapshot-{name}")
            passes[name] = {"workload": {**record, "result": stdout.strip().splitlines()[-1] if stdout.strip() else None},
                            "consumer": consumer_record, "snapshot": snapshot_record,
                            "exports": exported + snapshot_exported, "files": sizes,
                            "snapshot_files": snapshot_sizes}
            volume.commit()
        steps.finish("service-" + ("full" if summary["service_mode"] == "full" else "reduced"), argv, service, 60,
                     stop=True)
    else:
        steps.finish("service-reduced", argv, service, 30, stop=True)
    summary["passes"] = passes

    # Nsight Systems on the same case, after the SDK service has stopped.
    nsys_out = out / "nsys-trace"
    record, stdout = steps.run("nsys-diagnose", ["python", "-m", "utils.diagnose", "nsys", "--backend", backend,
                                                 "--workdir", str(workdir), "--out", str(nsys_out),
                                                 "--timeout-s", "900"], 1000, env=env)
    summary["nsys"] = {**record, "result": stdout.strip().splitlines()[-1] if stdout.strip() else None}
    summary["steps"] = steps.records
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return {key: summary[key] for key in ("run_id", "gpu_query", "driver_version", "service_mode",
                                           "workload_alone", "nsys") if key in summary} | {
        "passes": {name: {"workload_rc": p["workload"]["returncode"], "consumer_rc": p["consumer"]["returncode"],
                          "files": len(p["files"])} for name, p in passes.items()}}


@app.local_entrypoint()
def main(run_id: str = "", task: str = "relu_squared", case: str = "b4096_h14336_bfloat16",
         backend: str = "triton", iters: int = 20000, duration_ms: int = 30000):
    run_id = run_id or "nsight-perf-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    call = trace.spawn(run_id, task, case, backend, iters, duration_ms)
    print(json.dumps({"run_id": run_id, "function_call_id": call.object_id, "gpu": GPU,
                      "volume": "agenticrl-cuda-artifacts", "remote": f"/artifacts/{run_id}"}), flush=True)
    print(json.dumps(wait_for_call(call), allow_nan=False, default=str), flush=True)
