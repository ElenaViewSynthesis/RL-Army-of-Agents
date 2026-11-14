import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from utils.diagnose import diagnostic_environment, run_process


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(source, output, importer, recovery_run_id):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / "manifest.json").read_text())
    rows = [row for row in json.loads((source / "records.json").read_text())["rows"]
            if row["tool"] == "nsys" and not row["negative_control"]]
    if not rows:
        raise ValueError("No recorded Nsight Systems captures")
    output.mkdir(parents=True, exist_ok=False)
    env = diagnostic_environment(output, "nsys")
    nsys = shutil.which("nsys")
    if nsys is None or not Path(importer).is_file():
        raise ValueError("Nsight Systems and its installed importer are required")
    versions = []
    for binary, label in ((str(importer), "importer_version"), (nsys, "nsys_version")):
        record = run_process([binary, "--version"], output, env, time.monotonic() + 30, output, label)
        versions.append(record)
        if record["returncode"] != 0 or record["timed_out"]:
            raise RuntimeError("Installed Nsight importer/version check failed")
    manifest = {"schema": "gpu-diagnostic-matrix-v1", "run_id": original["run_id"],
                "recovery_run_id": recovery_run_id, "diagnostic_only": True,
                "source_directory": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
                "planned": [["nsys", row["backend"], None, False] for row in rows],
                "scope": [row["backend"] for row in rows], "tools": ["nsys"], "checks": [],
                "versions": versions, "recovery_script_sha256": sha256(__file__),
                "gpu_recaptured": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    recovered = []
    for row in rows:
        relative = Path(row["report"])
        original_case = (source / relative).resolve().parent
        if not original_case.is_relative_to(source) or relative.name != "report.json":
            raise ValueError("Capture report must remain inside the source directory")
        case = output / relative.parent
        case.mkdir(parents=True)
        for name in ("report.json", "worker.json", "gpu.json", "tool.stdout.log", "tool.stderr.log",
                     "version.stdout.log", "version.stderr.log", "capture.qdstrm"):
            shutil.copyfile(original_case / name, case / name)
        shutil.copyfile(original_case / "capture.qdstrm", case / "original.qdstrm")
        env = diagnostic_environment(case, "nsys")
        deadline = time.monotonic() + 600
        stages = [run_process([str(importer), "--input-file", str(case / "capture.qdstrm"),
                               "--output-file", str(case / "capture.nsys-rep")],
                              case, env, deadline, case, "import")]
        if stages[-1]["returncode"] == 0 and not stages[-1]["timed_out"]:
            stages.append(run_process([nsys, "export", "--type=sqlite", "--output",
                                       str(case / "capture.sqlite"), str(case / "capture.nsys-rep")],
                                      case, env, deadline, case, "export"))
        complete = (len(stages) == 2 and all(stage["returncode"] == 0 and not stage["timed_out"]
                                            for stage in stages)
                    and all((case / name).is_file() and (case / name).stat().st_size > 0
                            for name in ("capture.nsys-rep", "capture.sqlite")))
        receipt = {"source_gpu_run_id": original["run_id"], "recovery_run_id": recovery_run_id,
                   "gpu_recaptured": False, "complete": complete, "stages": stages,
                   "source_report_sha256": sha256(original_case / "report.json"),
                   "source_qdstrm_sha256": sha256(original_case / "capture.qdstrm"),
                   "processed_qdstrm_sha256": sha256(case / "capture.qdstrm"),
                   "versions": versions, "recovery_script_sha256": manifest["recovery_script_sha256"],
                   "output_sha256": {name: sha256(case / name) for name in
                                     ("capture.nsys-rep", "capture.sqlite") if (case / name).is_file()}}
        (case / "postprocess.json").write_text(json.dumps(receipt, indent=2) + "\n")
        recovered.append(row | {"postprocessed": complete})
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(recovered[-1]) + "\n")
        print(json.dumps({"case": row["case"], "postprocessed": complete}), flush=True)
    (output / "records.json").write_text(json.dumps(
        {"run_id": original["run_id"], "recovery_run_id": recovery_run_id,
         "diagnostic_only": True, "rows": recovered}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--importer", type=Path, required=True)
    parser.add_argument("--recovery-run-id", required=True)
    args = parser.parse_args()
    recover(args.input_dir, args.output_dir, args.importer, args.recovery_run_id)


if __name__ == "__main__":
    main()
