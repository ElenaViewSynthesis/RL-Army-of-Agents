"""Opt-in W&B logging of progressive task-suite evidence (one run per suite run).

Reads a downloaded evidence folder (summary.json, suite.json or
suite/suite.json, request.json, source-manifest.json) and logs it as one W&B
run: config (request, frozen-source hash, git revision), a `suite_rows`
table, summary counts, and the JSON/log evidence as an artifact. Suite rows
are evaluation records, not training metrics, so nothing goes into metric
history beyond one iteration-0 snapshot. No SDK import until logging;
`--dry-run` needs no SDK. See Architecture-decision/0002-wandb-tracking-contract.md.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

from tracking.wandb_log import ENTITY, PROJECT, _table, define_metrics

DROPPED = ("log/", "artifact/")
EVIDENCE_SUFFIXES = (".json", ".jsonl", ".log")


def _load(path):
    return json.loads(Path(path).read_text())


def _suite(folder):
    for candidate in (folder / "suite" / "suite.json", folder / "suite.json"):
        if candidate.is_file():
            return _load(candidate)
    raise FileNotFoundError(f"{folder}: no suite.json")


def suite_payload(folder):
    """Build the W&B payload for one evidence folder; raises if it is incomplete."""
    folder = Path(folder)
    summary, request, suite = _load(folder / "summary.json"), _load(folder / "request.json"), _suite(folder)
    manifest_path = folder / "source-manifest.json"
    manifest = _load(manifest_path) if manifest_path.is_file() else {}
    rows = [{k: v for k, v in row.items() if not k.startswith(DROPPED)} for row in suite["rows"]]
    counts = suite["counts"]
    if sum(counts.values()) != len(rows):
        raise ValueError(f"{folder}: counts {counts} do not match {len(rows)} rows")
    run_id = request["run_id"]
    files = sorted(str(p) for p in folder.rglob("*")
                   if p.is_file() and p.suffix in EVIDENCE_SUFFIXES)
    return {
        "run_id": run_id,
        "wandb_id": f"h200-{run_id}".replace("/", "-")[:64],
        "name": run_id,
        "kind": "task-suite-h200",
        "tags": ["task-suite", "h200", f"case-set:{request['case_set']}",
                 *sorted({f"stage{row['suite/stage']}" for row in suite["rows"]})],
        "notes": ("Progressive task suite on Modal H200. Name-based kernel attribution; "
                  "benchmark rates are diagnostics, not reward inputs."),
        "config": {"suite/request": request,
                   "suite/case_set": request["case_set"],
                   "suite/backends": request["backends"],
                   "suite/fixtures": request["fixtures"],
                   "suite/tasks": request["tasks"],
                   "suite/gpu_requested": request.get("gpu_requested"),
                   "source/manifest_sha256": (hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                                              if manifest_path.is_file() else None),
                   "source/git_revision": manifest.get("git_revision"),
                   "source/files": len(manifest.get("files", {})) or None},
        "metrics": {f"suite/{k}": v for k, v in counts.items()},
        "summary": {"suite/status": summary["status"],
                    "suite/source_mismatches": len(summary.get("source_mismatches") or []),
                    "suite/wall_s": summary.get("execution", {}).get("wall_s"),
                    **{f"suite/{k}": v for k, v in counts.items()}},
        "rows": rows,
        "files": files,
        "root": str(folder),
    }


def log_suite(payload, entity=ENTITY, project=PROJECT, wandb_module=None):
    wandb = wandb_module or importlib.import_module("wandb")
    run = wandb.init(entity=entity, project=project, id=payload["wandb_id"], name=payload["name"],
                     job_type=payload["kind"], tags=payload["tags"], notes=payload["notes"],
                     config=payload["config"], resume="allow")
    define_metrics(run)
    run.log({"iteration": 0, **payload["metrics"],
             "suite_rows": _table(wandb, payload["rows"], ["suite/task"])})
    run.summary.update(payload["summary"])
    artifact = wandb.Artifact(name=payload["wandb_id"], type="task-suite-run",
                              metadata={"status": payload["summary"]["suite/status"],
                                        "rows": len(payload["rows"])})
    root = Path(payload["root"])
    for path in payload["files"]:
        artifact.add_file(path, name=str(Path(path).relative_to(root)))
    run.log_artifact(artifact)
    run.finish(exit_code=0 if payload["summary"]["suite/status"] == "pass" else 1)
    return getattr(run, "url", run.id)


def readback(payload, entity=ENTITY, project=PROJECT, api=None, attempts=18, delay_s=10.0, sleep=None):
    """Return a list of mismatches between the logged run and the payload (empty = OK).

    The W&B API served a just-finished run's summary after one to two
    minutes (2026-11-10), so a mismatch is re-read with a fresh API client up
    to `attempts` times, `delay_s` apart (three minutes by default).
    """
    import time
    sleep = sleep or time.sleep
    problems = []
    for attempt in range(attempts):
        client = api or importlib.import_module("wandb").Api()
        run = client.run(f"{entity}/{project}/{payload['wandb_id']}")
        problems = [key for key, value in payload["summary"].items()
                    if key != "suite/wall_s" and run.summary.get(key) != value]
        for key in ("source/manifest_sha256", "source/git_revision"):
            if run.config.get(key) != payload["config"][key]:
                problems.append(key)
        if not problems:
            return []
        if attempt + 1 < attempts:
            sleep(delay_s)
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    args = parser.parse_args(argv)
    payloads = [suite_payload(folder) for folder in args.folders]
    if args.dry_run:
        print(json.dumps([{k: (len(v) if k in ("rows", "files") else v) for k, v in p.items()}
                          for p in payloads], indent=2, allow_nan=False))
        return
    for payload in payloads:
        url = log_suite(payload, args.entity, args.project)
        problems = readback(payload, args.entity, args.project)
        print(json.dumps({"run_id": payload["run_id"], "url": url, "readback_ok": not problems,
                          "readback_mismatches": problems}))
        if problems:
            raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
