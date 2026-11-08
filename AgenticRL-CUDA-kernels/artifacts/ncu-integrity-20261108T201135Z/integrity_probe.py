"""CPU-only remote sha256 of the run-20261108T193605Z volume dir; no GPU."""
import json
import time
from pathlib import Path

import modal

from cloud.modal_job import wait_for_call
from cloud.modal_runtime import image, volume

app = modal.App("agenticrl-ncu-integrity", image=image)


@app.function(timeout=1800, volumes={"/artifacts": volume})
def ncu_integrity(run_id: str) -> dict:
    import hashlib
    root = Path("/artifacts/20261108T193605Z")
    data = {"source_run_id": "20261108T193605Z",
            "sha256_by_relative_path": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(root.rglob("*")) if p.is_file()}}
    out = Path("/artifacts") / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "ncu-integrity.json").write_text(json.dumps(data, indent=2) + "\n")
    volume.commit()
    return data


@app.local_entrypoint()
def main():
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(json.dumps(wait_for_call(ncu_integrity.spawn(run_id))), flush=True)
