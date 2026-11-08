"""Modal cloud-GPU runtime for the AgenticRL CUDA kernels environment.

`modal run cloud/modal_runtime.py` runs the runtime preflight on a GPU host.
`modal run cloud/modal_runtime.py --tests` additionally runs, in order: the
--require profile gate (recorded, not fatal), the stock pipeline,
scripts.record_kernel_names, the GPU unittest suites, and scripts.validate.
MODAL_GPU selects the GPU (default "H200"). Outputs land in the Modal Volume
`agenticrl-cuda-artifacts` mounted at /artifacts, under a UTC run_id
directory.
"""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import modal
from cloud.acceptance import review
from utils.backend import BACKENDS

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/AgenticRL-CUDA-kernels"
GPU = os.environ.get("MODAL_GPU", "H200")

image = (
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("ninja-build", "git")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0", "PYTHONPATH": REMOTE})
    .add_local_dir(
        str(ROOT), remote_path=REMOTE,
        ignore=[".venv", ".cache", "build", "__pycache__", "artifacts",
                ".env", ".env.*", "*.so", "*.pyc"])
    .add_local_file(str(ROOT / "artifacts/upstream.json"),
                    remote_path=f"{REMOTE}/artifacts/upstream.json")
)

volume = modal.Volume.from_name("agenticrl-cuda-artifacts", create_if_missing=True)
app = modal.App("agenticrl-cuda-kernels", image=image)

DIAGNOSTIC_ENV = (
    "TRITON_KERNEL_DUMP", "TRITON_DUMP_DIR", "TRITON_ALWAYS_COMPILE", "MLIR_ENABLE_DUMP",
    "LLVM_IR_ENABLE_DUMP", "TRITON_PRINT_AUTOTUNING", "TILELANG_PRINT_ON_COMPILATION",
    "CUDA_TILE_LOGS", "CUDA_TILE_DUMP_BYTECODE", "CUTE_DSL_PRINT_IR", "CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR",
)


def forwarded_environment(enabled):
    return {k:os.environ[k] for k in DIAGNOSTIC_ENV if k in os.environ} if enabled else {}


def await_job(function, *args):
    """Durable invocation; polling never resubmits a measurement job."""
    call = function.spawn(*args)
    print(json.dumps({"function_call_id": call.object_id}), flush=True)
    while True:
        try:
            return call.get(timeout=30)
        except TimeoutError:
            pass


def _run(args, name, timeout_s, run_id, diagnostic_env=None):
    # A hung step is recorded, not raised: its partial output still reaches the volume.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(diagnostic_env or {})}
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(args, cwd=REMOTE, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out, returncode = True, None
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    elapsed = time.monotonic() - started
    out_dir = Path("/artifacts") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.stdout.log").write_text(stdout)
    (out_dir / f"{name}.stderr.log").write_text(stderr)
    volume.commit()
    return {"cmd": shlex.join(args), "returncode": returncode, "timed_out": timed_out,
            "wall_s": elapsed,
            "stdout_tail": stdout[-20000:], "stderr_tail": stderr[-8000:]}


@app.function(gpu=GPU, timeout=1800, volumes={"/artifacts": volume})
def preflight(run_id: str) -> dict:
    record = _run(["python", "-m", "utils.runtime", "--json"], "preflight", 300, run_id)
    smi = _run(["nvidia-smi"], "nvidia_smi", 60, run_id)
    try:
        parsed = json.loads(record["stdout_tail"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        parsed = {"runtime/available": False, "preflight_stdout": record}
    return {**parsed, "nvidia_smi": smi["stdout_tail"], "run_id": run_id}


@app.function(gpu=GPU, timeout=5400, volumes={"/artifacts": volume})
def gpu_tests(run_id: str, backends: str = "", diagnostic_env: dict = None) -> dict:
    selected = backends.split(",") if backends else list(BACKENDS)
    out_dir = Path("/artifacts") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove uploaded stale reports before starting. Failed stages cannot package history.
    for name in ("results.json", "kernel_names.json"):
        (Path(REMOTE) / "artifacts" / name).unlink(missing_ok=True)
    targets = ["tests.test_provenance", "tests.test_dsl_kernels", "tests.test_benchmark.GPUBenchmarkTests"]
    if set(selected) != set(BACKENDS):
        targets = []
        if {"cuda", "triton"} <= set(selected):
            targets.append("tests.test_provenance.GPUIntegrationTests")
        if "triton" in selected:
            targets += ["tests.test_provenance.CPUIntegrationTests", "tests.test_benchmark.GPUBenchmarkTests"]
        for backend, cls in (("tilelang","TileLang"),("cutile","CuTile"),("cutedsl","CuTeDSL")):
            if backend in selected:
                targets += [f"tests.test_provenance.{cls}GPUIntegrationTests", f"tests.test_dsl_kernels.{cls}AxpbyTests"]
    commands = [
        # A non-zero exit here is recorded, not fatal: it names the DSL the
        # image failed to provide while the other suites still run.
        (["python", "-m", "utils.runtime", "--json", "--require", ",".join(selected)], "require", 300),
        *([(["bash", "-c",
              "bash utils/compile.sh && python -m utils.verification && "
              "python -m utils.profiling --json"], "stock_pipeline", 1800)] if "cuda" in selected else []),
        (["python", "-m", "scripts.record_kernel_names", "--device", "cuda", "--backends", ",".join(selected)],
             "kernel_names", 1800),
        *([(["python", "-m", "unittest", "-v", *targets], "gpu_tests", 3600)] if targets else []),
        (["python", "-m", "scripts.validate", "--device", "cuda", "--backends", ",".join(selected)], "validate", 3600),
    ]
    stage_env = {**(diagnostic_env or {}), "CUDA_AGENT_BENCH_SAMPLES_DIR": str(out_dir / "benchmark-samples"),
                 "CUDA_AGENT_STAGE_TIMINGS": "1"}
    runs = [_run(args, name, timeout, run_id, stage_env) for args, name, timeout in commands]
    for name in ("results.json", "kernel_names.json"):
        source = Path(REMOTE) / "artifacts" / name
        if source.is_file():
            (out_dir / name).write_bytes(source.read_bytes())
    # The uploaded artifacts/environment.json may describe local CPU development.
    # Use the require probe from this container, never that historical snapshot.
    try:
        environment = json.loads((out_dir / "require.stdout.log").read_text().strip().splitlines()[-1])
    except (ValueError, IndexError):
        environment = {"runtime/available": False, "reason": "require probe emitted no JSON"}
    (out_dir / "environment.json").write_text(json.dumps(
        {"run_id": run_id, "command": runs[0]["cmd"], **environment}, indent=2))
    def load(name):
        try:
            return json.loads((out_dir / name).read_text())
        except (OSError, ValueError):
            return {}
    try:
        stock = json.loads((out_dir / "stock_pipeline.stdout.log").read_text().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        stock = {}
    acceptance = review(run_id, selected, runs, load("results.json"), load("kernel_names.json"),
                        load("environment.json"), environment, stock)
    acceptance["diagnostic_env"] = diagnostic_env or {}
    (out_dir / f"acceptance-review-{run_id}.json").write_text(json.dumps(acceptance, indent=2, allow_nan=False)+"\n")
    volume.commit()
    return {"run_id": run_id, "runs": runs, "acceptance": acceptance["status"]}


@app.function(gpu=GPU, timeout=21600, volumes={"/artifacts": volume})
def diagnose_timing(run_id: str, backends: str = "", repeat: int = 0, diagnostic_env: dict = None) -> dict:
    require = _run(["python", "-m", "utils.runtime", "--json", "--require", backends or ",".join(BACKENDS)],
                   "require", 300, run_id, diagnostic_env)
    args = ["python", "-m", "scripts.diagnose_timing", "--output-dir", f"/artifacts/{run_id}/timing",
            "--backends", backends or ",".join(BACKENDS)]
    if repeat:
        args += ["--repeat", str(repeat)]
    # One measurement process at a time; the parent only reports progress.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    progress = Path(f"/artifacts/{run_id}/timing/progress.jsonl")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run, args, "diagnose_timing", 21000, run_id, diagnostic_env)
        while True:
            try:
                record = future.result(timeout=30)
                break
            except TimeoutError:
                try:
                    lines = progress.read_text().splitlines()
                    last = json.loads(lines[-1])
                    print(json.dumps({"run_id": run_id, "completed": len(lines),
                                      "backend": last["backend"], "repeat": last["repeat"],
                                      "reason": last.get("reason") or last.get("profile", {}).get("timing_reason")}), flush=True)
                except (OSError, ValueError, IndexError):
                    pass
    return {"run_id": run_id, "runs": [require, record]}


@app.local_entrypoint()
def main(tests: bool = False, diagnose: bool = False, backends: str = "", repeat: int = 0,
         forward_diagnostics: bool = False):
    # One run_id per invocation, shared by both remote calls even across containers.
    selected = backends.split(",") if backends else list(BACKENDS)
    if any(b not in BACKENDS for b in selected) or repeat < 0:
        raise ValueError("known --backends and nonnegative --repeat required")
    if tests and diagnose:
        raise ValueError("choose --tests or --diagnose")
    diagnostics = forwarded_environment(forward_diagnostics)
    for _ in range(1 if diagnose else (repeat or 1)):
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        print(json.dumps(preflight.remote(run_id), indent=2), flush=True)
        if diagnose:
            print(json.dumps(await_job(diagnose_timing, run_id, ",".join(selected), repeat, diagnostics), indent=2), flush=True)
        elif tests:
            print(json.dumps(await_job(gpu_tests, run_id, ",".join(selected), diagnostics), indent=2), flush=True)
