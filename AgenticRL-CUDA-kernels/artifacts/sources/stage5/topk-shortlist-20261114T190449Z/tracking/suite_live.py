"""Opt-in live W&B progress for a Modal task-suite run, watched from the local machine.

The Modal runner appends one JSON line per finished row to
`/<run_id>/suite/progress.jsonl` on the `agenticrl-cuda-artifacts` volume and
commits the volume every 30 s. This watcher polls that file with the local
`modal` CLI, logs each new row to the W&B run `h200-<run_id>` as progress
metrics, and stops when the run's `summary.json` appears. With `--finalize`, it
then downloads the run, copies the evidence subset (JSON, JSONL, logs,
Markdown and `source/task-source-manifest.json`) into `artifacts/runs/<run_id>/`
and hands that folder to `tracking.suite_log`, which logs the `suite_rows`
table, summary and evidence artifact into the same W&B run.

Nothing here touches the GPU job, the Modal image or the frozen source; W&B
credentials stay local. Progress metrics are diagnostics stepped by the row
count, `live/row`; they are never reward inputs. See ADR 0002, section
"Live task-suite progress". No SDK import until logging; `--dry-run` needs
no SDK and prints what would be logged.
"""

import argparse
import importlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tracking.wandb_log import ENTITY, PROJECT

VOLUME = "agenticrl-cuda-artifacts"
STATUSES = ("passed", "failed", "unavailable")
# Row result key -> live metric name. Only numeric scalars; strings stay in the final table.
ROW_METRICS = {"reward/value": "live/row_reward_value",
               "env/t_gen_us": "live/row_t_gen_us",
               "env/t_eager_us": "live/row_t_eager_us",
               "env/t_compile_us": "live/row_t_compile_us",
               "env/provenance_agent": "live/row_provenance_agent",
               "suite/wall_s": "live/row_wall_s"}
EVIDENCE_SUFFIXES = (".json", ".jsonl", ".log", ".md")


def wandb_id(run_id):
    return f"h200-{run_id}".replace("/", "-")[:64]


def parse_progress(text):
    """Complete JSON lines only; a partly written last line is left for the next poll."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            break
    return rows


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def row_metrics(index, row, counts):
    """Metrics for the `index`-th finished row (1-based); `counts` are cumulative after it."""
    metrics = {"live/row": index, "live/completed": index,
               **{f"live/{status}": counts[status] for status in STATUSES},
               "live/row_status_passed": int(row.get("suite/status") == "passed")}
    for key, name in ROW_METRICS.items():
        value = _number(row.get(key))
        if value is not None:
            metrics[name] = value
    return metrics


def row_label(row):
    return "/".join(str(row.get(k)) for k in ("suite/task", "suite/case", "suite/backend", "suite/fixture"))


class ModalVolume:
    """Reads single files from the run folder on the Modal volume through the `modal` CLI."""

    def __init__(self, run_id, modal="modal", volume=VOLUME, runner=subprocess.run):
        self.run_id, self.modal, self.volume, self.runner = run_id, modal, volume, runner

    def read(self, relative):
        """Return the file's text, or None while it does not exist (or the CLI fails)."""
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "file"
            proc = self.runner([self.modal, "volume", "get", "--force", self.volume,
                                f"/{self.run_id}/{relative}", str(target)],
                               capture_output=True, text=True, timeout=300)
            if proc.returncode != 0 or not target.is_file():
                return None
            return target.read_text()

    def download(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        proc = self.runner([self.modal, "volume", "get", "--force", self.volume,
                            f"/{self.run_id}/", str(destination)],
                           capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"modal volume get failed: {proc.stderr[-500:]}")
        return destination / self.run_id


def collect_evidence(downloaded, destination, client_log=None):
    """Copy the evidence subset of a downloaded run; never overwrite an existing folder."""
    downloaded, destination = Path(downloaded), Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} exists; refusing to overwrite evidence")
    copied = []
    for path in sorted(downloaded.rglob("*")):
        relative = path.relative_to(downloaded)
        parts = relative.parts
        if (not path.is_file() or path.suffix not in EVIDENCE_SUFFIXES or "source" in parts[:-1]
                or "build" in parts or "cache" in parts or any(p.startswith(".") for p in parts[:-1])):
            continue
        copied.append(relative)
    manifest = downloaded / "source" / "task-source-manifest.json"
    if manifest.is_file():
        copied.append(manifest.relative_to(downloaded))
    for relative in copied:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded / relative, target)
    if client_log and Path(client_log).is_file():
        shutil.copyfile(client_log, destination / "modal-session.log")
    return destination


class LiveSuite:
    """One live W&B session for one suite run; rows are logged once, in file order."""

    def __init__(self, run_id, source, entity=ENTITY, project=PROJECT, wandb_module=None, dry_run=False,
                 out=sys.stdout):
        self.run_id, self.source, self.entity, self.project = run_id, source, entity, project
        self.wandb_module, self.dry_run, self.out = wandb_module, dry_run, out
        self.run, self.logged, self.started = None, 0, False
        self.counts = {status: 0 for status in STATUSES}

    def _start(self, request):
        config = {"suite/request": request, "suite/case_set": request.get("case_set"),
                  "suite/backends": request.get("backends"), "suite/fixtures": request.get("fixtures"),
                  "suite/tasks": request.get("tasks"), "suite/gpu_requested": request.get("gpu_requested")}
        if self.dry_run:
            self._print({"event": "start", "wandb_id": wandb_id(self.run_id), "config": config})
            return
        wandb = self.wandb_module or importlib.import_module("wandb")
        self.wandb_module = wandb
        self.run = wandb.init(entity=self.entity, project=self.project, id=wandb_id(self.run_id),
                              name=self.run_id, job_type="task-suite-h200",
                              tags=["task-suite", "h200", "live", f"case-set:{request.get('case_set')}"],
                              notes=("Progressive task suite on Modal H200, logged live by "
                                     "tracking.suite_live. Name-based kernel attribution; progress "
                                     "metrics are diagnostics, not reward inputs."),
                              config=config, resume="allow")
        self.run.define_metric("live/row", summary="none")
        self.run.define_metric("live/*", step_metric="live/row", summary="none")

    def _print(self, payload):
        print(json.dumps(payload, allow_nan=False), file=self.out, flush=True)

    def poll(self):
        """Log rows that are new since the last poll; return the number logged."""
        if not self.started:
            text = self.source.read("request.json")
            if text is None:
                return 0
            self._start(json.loads(text))
            self.started = True
        rows = parse_progress(self.source.read("suite/progress.jsonl") or "")
        new = rows[self.logged:]
        for row in new:
            self.logged += 1
            status = row.get("suite/status")
            if status in self.counts:
                self.counts[status] += 1
            metrics = row_metrics(self.logged, row, self.counts)
            if self.dry_run:
                self._print({"event": "row", "label": row_label(row), "metrics": metrics})
            else:
                self.run.log(metrics)
            self._print({"completed": self.logged, "row": row_label(row), "status": status})
        return len(new)

    def finished(self):
        text = self.source.read("summary.json")
        return json.loads(text) if text else None

    def finish(self, summary):
        status = (summary or {}).get("status", "unknown")
        if self.run is not None:
            self.run.finish(exit_code=0 if status == "pass" else 1)
        self._print({"event": "finished", "status": status, "rows": self.logged, **self.counts})


def watch(live, interval_s=60.0, max_wait_s=6 * 3600.0, sleep=time.sleep, clock=time.monotonic):
    """Poll until summary.json exists or `max_wait_s` passes; return the summary or None."""
    started = clock()
    while True:
        live.poll()
        summary = live.finished()
        if summary is not None:
            live.poll()  # rows committed together with the summary
            live.finish(summary)
            return summary
        if clock() - started >= max_wait_s:
            live.finish(None)
            return None
        sleep(interval_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--max-wait-s", type=float, default=6 * 3600.0)
    parser.add_argument("--modal", default="modal", help="path to the modal CLI")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--dry-run", action="store_true", help="print metrics; no W&B contact")
    parser.add_argument("--finalize", action="store_true",
                        help="after the run: download, copy evidence to artifacts/runs/<run_id>, log it with suite_log")
    parser.add_argument("--evidence-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--client-log", type=Path, help="local `modal run` log to keep as modal-session.log")
    args = parser.parse_args(argv)
    source = ModalVolume(args.run_id, modal=args.modal)
    live = LiveSuite(args.run_id, source, args.entity, args.project, dry_run=args.dry_run)
    summary = watch(live, args.interval_s, args.max_wait_s)
    if summary is None:
        raise SystemExit("summary.json did not appear within --max-wait-s; nothing finalized")
    if not args.finalize:
        return
    from tracking import suite_log
    with tempfile.TemporaryDirectory() as scratch:
        downloaded = source.download(scratch)
        folder = collect_evidence(downloaded, args.evidence_root / args.run_id, args.client_log)
    payload = suite_log.suite_payload(folder)
    if args.dry_run:
        print(json.dumps({"event": "finalize-dry-run", "folder": str(folder), "rows": len(payload["rows"])}))
        return
    url = suite_log.log_suite(payload, args.entity, args.project)
    problems = suite_log.readback(payload, args.entity, args.project)
    print(json.dumps({"run_id": args.run_id, "url": url, "evidence": str(folder),
                      "readback_ok": not problems, "readback_mismatches": problems}))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
