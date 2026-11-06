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

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/AgenticRL-CUDA-kernels"
GPU = os.environ.get("MODAL_GPU", "H200")

image = (
    modal.Image.from_registry("nvidia/cuda:13.3.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("ninja-build", "git")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .add_local_dir(
        str(ROOT), remote_path=REMOTE,
        ignore=[".venv", ".cache", "build", "__pycache__", "artifacts/tmp",
                ".env", ".env.*", "*.so", "*.pyc"])
)

volume = modal.Volume.from_name("agenticrl-cuda-artifacts", create_if_missing=True)
app = modal.App("agenticrl-cuda-kernels", image=image)

def _run(args, name, timeout_s, run_id):
    # A hung step is recorded, not raised: its partial output still reaches the volume.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    timed_out = False
    try:
        proc = subprocess.run(args, cwd=REMOTE, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out, returncode = True, None
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    out_dir = Path("/artifacts") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.stdout.log").write_text(stdout)
    (out_dir / f"{name}.stderr.log").write_text(stderr)
    volume.commit()
    return {"cmd": shlex.join(args), "returncode": returncode, "timed_out": timed_out,
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
def gpu_tests(run_id: str) -> dict:
    runs = [
        # A non-zero exit here is recorded, not fatal: it names the DSL the
        # image failed to provide while the other suites still run.
        _run(["python", "-m", "utils.runtime", "--json", "--require",
              "cuda,triton,tilelang,cutile,cutedsl"], "require", 300, run_id),
        _run(["bash", "-c",
              "bash utils/compile.sh && python -m utils.verification && "
              "python -m utils.profiling --json"], "stock_pipeline", 1800, run_id),
        _run(["python", "-m", "scripts.record_kernel_names", "--device", "cuda"],
             "kernel_names", 1800, run_id),
        _run(["python", "-m", "unittest", "-v", "tests.test_provenance",
              "tests.test_dsl_kernels", "tests.test_benchmark.GPUBenchmarkTests"],
             "gpu_tests", 3600, run_id),
        _run(["python", "-m", "scripts.validate", "--device", "cuda"], "validate", 3600, run_id),
    ]
    out_dir = Path("/artifacts") / run_id
    for name in ("results.json", "environment.json", "kernel_names.json"):
        source = Path(REMOTE) / "artifacts" / name
        if source.is_file():
            (out_dir / name).write_bytes(source.read_bytes())
    volume.commit()
    return {"run_id": run_id, "runs": runs}


@app.local_entrypoint()
def main(tests: bool = False):
    # One run_id per invocation, shared by both remote calls even across containers.
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(json.dumps(preflight.remote(run_id), indent=2))
    if tests:
        print(json.dumps(gpu_tests.remote(run_id), indent=2))
