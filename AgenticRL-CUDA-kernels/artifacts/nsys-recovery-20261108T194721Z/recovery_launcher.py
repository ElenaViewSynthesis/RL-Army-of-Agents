"""One-off CPU recovery of the five captured nsys QDSTRM traces; no GPU.

`modal run --detach recovery_launcher.py` runs the authored
scripts.recover_nsys importer + sqlite export on the fixed image and records
installed OS provider metadata under <run_id>/nsys-recovery.
"""

import json
import time
from pathlib import Path

import modal

from cloud.modal_job import wait_for_call
from cloud.modal_runtime import image, volume

app = modal.App("agenticrl-nsys-recovery", image=image)

IMPORTER = "/opt/nvidia/nsight-compute/2026.2.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter"
DPKG = ["dpkg-query", "-W", "-f=${binary:Package} ${Version}\n", "libdw1t64", "libelf1t64"]


@app.function(timeout=3600, volumes={"/artifacts": volume})
def nsys_recovery(run_id: str) -> dict:
    import subprocess
    run_dir = Path("/artifacts") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "nsys-recovery"
    started = time.monotonic()
    record = {"cmd": ["python", "-m", "scripts.diagnose_gpu"], "returncode": None}
    argv = ["python", "-m", "scripts.recover_nsys", "--input-dir",
            "/artifacts/20261108T192232Z/gpu-diagnostics", "--output-dir",
            str(out_dir), "--importer", IMPORTER, "--recovery-run-id", run_id]
    try:
        proc = subprocess.run(argv, cwd="/root/AgenticRL-CUDA-kernels",
                              capture_output=True, text=True, timeout=3300,
                              env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
        record = {"cmd": __import__("shlex").join(argv), "returncode": proc.returncode,
                  "timed_out": False, "wall_s": time.monotonic() - started,
                  "stdout_tail": proc.stdout[-20000:], "stderr_tail": proc.stderr[-8000:]}
    except subprocess.TimeoutExpired as exc:
        record = {"cmd": __import__("shlex").join(argv), "returncode": None,
                  "timed_out": True, "wall_s": time.monotonic() - started,
                  "stdout_tail": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
                  "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else ""}
    (run_dir / "nsys_recovery.stdout.log").write_text(record["stdout_tail"])
    (run_dir / "nsys_recovery.stderr.log").write_text(record["stderr_tail"])
    pkg = {"argv": DPKG}
    try:
        proc = subprocess.run(DPKG, capture_output=True, text=True, timeout=30)
        pkg["returncode"] = proc.returncode
        (run_dir / "dpkg.stdout.log").write_text(proc.stdout)
        (run_dir / "dpkg.stderr.log").write_text(proc.stderr)
    except OSError as exc:
        pkg["error_type"] = type(exc).__name__
    record["dpkg"] = pkg
    (run_dir / "recovery-meta.json").write_text(json.dumps(record, indent=2) + "\n")
    volume.commit()
    return {"run_id": run_id, "recovery": record}


@app.local_entrypoint()
def main():
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(json.dumps(wait_for_call(nsys_recovery.spawn(run_id)), indent=2), flush=True)
