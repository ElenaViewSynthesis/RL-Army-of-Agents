"""One-off CPU probe for the Nsight Systems QdstrmImporter; no GPU is allocated.

`modal run --detach importer_probe.py` resolves the installed `nsys`, searches
bounded vendor locations for QdstrmImporter binaries, and records raw
--version/--help/ldd output under <run_id>/nsys-importer on the shared volume.
"""

import json
import time
from pathlib import Path

import modal

from cloud.modal_job import wait_for_call
from cloud.modal_runtime import image, volume

app = modal.App("agenticrl-importer-probe", image=image)


@app.function(timeout=900, volumes={"/artifacts": volume})
def importer_probe(run_id: str) -> dict:
    import os
    import shutil
    import stat
    import subprocess
    out_dir = Path("/artifacts") / run_id / "nsys-importer"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"run_id": run_id, "nsys": {}, "importers": []}

    resolved = shutil.which("nsys")
    results["nsys"] = {"which": resolved,
                       "realpath": os.path.realpath(resolved) if resolved else None}

    roots = [Path("/opt/nvidia")]
    if results["nsys"]["realpath"]:
        roots.append(Path(results["nsys"]["realpath"]).parent.parent)
    found = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*QdstrmImporter*"):
            real = os.path.realpath(path)
            if path.is_file() and real not in seen:
                seen.add(real)
                found.append(path)
    results["search_roots"] = [str(r) for r in roots]

    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/root")}
    for index, path in enumerate(found):
        mode = stat.S_IMODE(path.stat().st_mode)
        record = {"path": str(path), "realpath": os.path.realpath(path),
                  "mode_octal": oct(mode), "commands": []}
        for argv in ([str(path), "--version"], [str(path), "--help"], ["ldd", str(path)]):
            label = f"{index}-" + Path(argv[0]).name + "-" + argv[1].lstrip("-")
            entry = {"argv": argv}
            try:
                proc = subprocess.run(argv, capture_output=True, text=True,
                                      timeout=30, env=env)
                entry["returncode"] = proc.returncode
                (out_dir / f"{label}.stdout.log").write_text(proc.stdout)
                (out_dir / f"{label}.stderr.log").write_text(proc.stderr)
                entry["stderr_head"] = proc.stderr[:500]
            except subprocess.TimeoutExpired:
                entry.update(timed_out=True, returncode=None)
            except OSError as exc:
                entry.update(error_type=type(exc).__name__, returncode=None)
            record["commands"].append(entry)
        text = ""
        version_err = out_dir / f"{index}-{Path(str(path)).name}-version.stderr.log"
        if version_err.is_file():
            text = version_err.read_text(errors="replace").lower()
        if "qt" in text and ("plugin" in text or "display" in text or "xcb" in text):
            qt_env = {**env, "QT_QPA_PLATFORM": "offscreen"}
            try:
                proc = subprocess.run([str(path), "--version"], capture_output=True,
                                      text=True, timeout=30, env=qt_env)
                record["qt_offscreen_retry"] = {
                    "argv": [str(path), "--version"], "env_key": "QT_QPA_PLATFORM",
                    "returncode": proc.returncode}
                (out_dir / f"{index}-version-offscreen.stdout.log").write_text(proc.stdout)
                (out_dir / f"{index}-version-offscreen.stderr.log").write_text(proc.stderr)
            except subprocess.TimeoutExpired:
                record["qt_offscreen_retry"] = {"argv": [str(path), "--version"],
                                                "env_key": "QT_QPA_PLATFORM",
                                                "timed_out": True, "returncode": None}
        results["importers"].append(record)

    (out_dir / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
    volume.commit()
    return results


@app.local_entrypoint()
def main():
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(json.dumps(wait_for_call(importer_probe.spawn(run_id)), indent=2), flush=True)
