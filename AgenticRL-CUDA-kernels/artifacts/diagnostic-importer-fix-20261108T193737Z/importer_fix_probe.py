"""CPU probe: verify QdstrmImporter runs after the libdw1 image fix; no GPU.

`modal run --detach importer_fix_probe.py` records ldd/PATH presence, dynamic
loader --list output, and raw --version/--help under <run_id>/importer-fix.
"""

import json
import time
from pathlib import Path

import modal

from cloud.modal_job import wait_for_call
from cloud.modal_runtime import image, volume

app = modal.App("agenticrl-importer-fix-probe", image=image)

IMPORTER = "/opt/nvidia/nsight-compute/2026.2.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter"
COMMANDS = [
    ["test", "-x", "/usr/bin/ldd"],
    ["printenv", "PATH"],
    ["/lib64/ld-linux-x86-64.so.2", "--list", IMPORTER],
    [IMPORTER, "--version"],
    [IMPORTER, "--help"],
    ["dpkg", "-s", "libdw1"],
]


@app.function(timeout=900, volumes={"/artifacts": volume})
def fix_probe(run_id: str) -> dict:
    import subprocess
    out_dir = Path("/artifacts") / run_id / "importer-fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, argv in enumerate(COMMANDS):
        entry = {"argv": argv}
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            entry["returncode"] = proc.returncode
            (out_dir / f"{i}-{Path(argv[0]).name}.stdout.log").write_text(proc.stdout)
            (out_dir / f"{i}-{Path(argv[0]).name}.stderr.log").write_text(proc.stderr)
        except (subprocess.TimeoutExpired, OSError) as exc:
            entry.update(error_type=type(exc).__name__, returncode=None)
        results.append(entry)
    (out_dir / "probe.json").write_text(json.dumps({"run_id": run_id, "commands": results}, indent=2) + "\n")
    volume.commit()
    return {"run_id": run_id, "commands": results}


@app.local_entrypoint()
def main():
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(json.dumps(wait_for_call(fix_probe.spawn(run_id)), indent=2), flush=True)
