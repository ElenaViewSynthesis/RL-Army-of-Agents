"""One-off CPU probe for installed profiling tool CLIs; no GPU is allocated.

`modal run --detach probe_launcher.py` runs a single CPU container on the
existing cloud.modal_runtime image and writes each command's raw
stdout/stderr/returncode under <run_id>/tool-probe on the shared volume.
"""

import json
import time
from pathlib import Path

import modal

from cloud.modal_job import wait_for_call
from cloud.modal_runtime import image, volume

app = modal.App("agenticrl-tool-probe", image=image)

COMMANDS = [
    ["compute-sanitizer", "--version"],
    ["compute-sanitizer", "--help"],
    ["nsys", "--version"],
    ["nsys", "profile", "--help"],
    ["nsys", "stats", "--help"],
    ["nsys", "export", "--help"],
    ["ncu", "--version"],
    ["ncu", "--help"],
]


@app.function(timeout=600, volumes={"/artifacts": volume})
def tool_probe(run_id: str) -> dict:
    import shutil
    import subprocess
    out_dir = Path("/artifacts") / run_id / "tool-probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for argv in COMMANDS:
        name = argv[0] + "-" + "-".join(a.lstrip("-") for a in argv[1:])
        record = {"cmd": argv, "resolved": shutil.which(argv[0])}
        if record["resolved"] is None:
            record.update(unavailable=True, returncode=None)
        else:
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
                record["returncode"] = proc.returncode
                (out_dir / f"{name}.stdout.log").write_text(proc.stdout)
                (out_dir / f"{name}.stderr.log").write_text(proc.stderr)
            except subprocess.TimeoutExpired:
                record.update(timed_out=True, returncode=None)
        results.append(record)
    (out_dir / "probe.json").write_text(
        json.dumps({"run_id": run_id, "commands": results}, indent=2) + "\n")
    volume.commit()
    return {"run_id": run_id, "commands": results}


@app.local_entrypoint()
def main():
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(json.dumps(wait_for_call(tool_probe.spawn(run_id)), indent=2), flush=True)
