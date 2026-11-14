"""Exercise the real W&B SDK with synthetic records; never contact W&B or Weave."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from tests.support import make_task
from utils.config import RolloutConfig
from utils.provenance import PROVENANCE_RULESET


SUMMARY_KEYS = {
    "cases", "verify_ok", "provenance_pass", "reward_positive", "reward_mean",
    "unmatched_agent_names_total", "backends_measured", "gpu_tests",
    "unavailable_backends",
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def snapshot(run, cfg, cases, expected_mean):
    check(run.offline, "SDK run is not offline")
    config = dict(run.config)
    check({k[4:]: v for k, v in config.items() if k.startswith("cfg/")}
          == cfg.to_dict(), "SDK config does not contain the complete preset")
    check(config["meta/provenance_rules"] == PROVENANCE_RULESET,
          "SDK config is missing the current classifier ID")
    summary = dict(run.summary)
    check(not (set(summary) - SUMMARY_KEYS - {k for k in summary if k.startswith("_")}),
          f"Unexpected summary fields: {sorted(summary)}")
    check(summary["cases"] == cases, "Summary case count is wrong")
    check(summary["reward_mean"] == expected_mean, "Summary reward mean is wrong")
    return {"run_id": run.id, "files_dir": run.dir,
            "offline": run.offline, "config": config, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="A new local evidence directory; existing paths are refused")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # All SDK storage stays in this evidence directory, without live credentials.
    for name in ("WANDB_API_KEY", "WANDB_SWEEP_ID", "WANDB_RUN_ID"):
        os.environ.pop(name, None)
    os.environ.update({
        "WANDB_MODE": "offline", "WANDB_ENTITY": "offline-smoke",
        "WANDB_PROJECT": "kernel-tracking-offline-smoke",
        "WANDB_BASE_URL": "http://127.0.0.1:9",
        "WANDB_DIR": str(output), "WANDB_CACHE_DIR": str(output / "cache"),
        "WANDB_DATA_DIR": str(output / "data"),
        "WANDB_CONFIG_DIR": str(output / "config"),
        "WANDB_DISABLE_CODE": "true", "WANDB_CONSOLE": "off",
    })
    import wandb
    from tracking import wandb_log

    cfg = dataclasses.replace(RolloutConfig.preset("cispo"), group_size=3,
                              run_name="synthetic-offline-sdk-smoke", seed=7)
    tasks_dir = output / "tasks"
    tasks_dir.mkdir()
    kinds = ("cuda", "triton", "reward-hack-torchop")
    tasks = [make_task(tasks_dir, kind) for kind in kinds]
    hashes = [hashlib.sha256((p / "model_new.py").read_bytes()).hexdigest()
              for p in tasks]
    rows, profiles = [], []
    for kind in kinds:
        honest = kind != "reward-hack-torchop"
        rows.append({
            "env/verify_ok": True, "env/provenance": float(honest),
            "env/t_eager_us": 100.0, "env/t_compile_us": 80.0,
            "env/t_gen_us": 78.0, "env/speedup_vs_compile": 80.0 / 78.0,
            "env/turns_used": 1, "env/wall_s": 0.0,
            "reward/value": 2.0 if honest else -1.0,
            "reward/base": 2 if honest else -1,
            "reward/reason": "faster_than_eager" if honest else "provenance",
            "reward/overlong_penalty": 0.0,
            "meta/backend": kind if honest else "triton",
            "meta/algo": "cispo", "meta/seed": 7, "meta/task_id": kind,
            "log/stderr_tail": "synthetic-stage-tail-not-a-table-column",
            "bench/available": False,
        })
        profiles.append({
            "timing_measured": True, "provenance/agent": float(honest),
            "provenance/library": 0.0, "provenance/torch": float(not honest),
            "provenance/memory": 0.0, "kernel_us/synthetic_event": 78.0,
        })
    environment = {"runtime/python": platform.python_version(),
                   "runtime/gpu_count": 0, "runtime/synthetic_smoke": True}
    report = {
        "purpose": "SDK serialization smoke test, not GPU or training evidence",
        "inputs": "All rewards, timings and profile shares below are synthetic",
        "wandb_version": importlib.metadata.version("wandb"),
        "classifier": PROVENANCE_RULESET, "candidate_sha256": hashes,
        "runs": [],
    }

    run = wandb_log.log_rollout(rows[0], cfg, profile=profiles[0], workdir=tasks[0],
                               environment=environment, step=0)
    try:
        run.summary.update({"case": "stale", "env/provenance": 0.0,
                            "reward/value": -1.0})
        wandb_log.log_rollout(rows[2], cfg, run=run, profile=profiles[2],
                             workdir=tasks[2], environment=environment, step=1)
        report["runs"].append(snapshot(run, cfg, 2, 0.5))
    finally:
        run.finish()

    run, _, _ = wandb_log.log_batch(
        ["axpby"] * 3, rows, cfg, step=7, profiles=profiles, workdirs=tasks,
        environment=environment)
    try:
        report["runs"].append(snapshot(run, cfg, 3, 1.0))
    finally:
        run.finish()

    # Read the SDK's actual table files, not the logger's in-memory row cache.
    expected = {h: p for h, p in zip(hashes, profiles)}
    for saved in report["runs"]:
        saved["tables"] = []
        for path in sorted(Path(saved["files_dir"]).rglob("*.table.json")):
            table = json.loads(path.read_text())
            records = [dict(zip(table["columns"], row)) for row in table["data"]]
            for row in records:
                sha = row["meta/model_new_sha256"]
                check(sha in expected, "Table contains an unexpected candidate hash")
                check(row["meta/provenance_rules"] == PROVENANCE_RULESET,
                      "Table classifier ID is missing")
                check(json.loads(row["case_config"]) == cfg.to_dict(),
                      "Table case config differs from the full preset")
                for bucket in wandb_log.SHARES:
                    check(row["env/provenance_" + bucket]
                          == expected[sha]["provenance/" + bucket],
                          f"Wrong {bucket} share for {sha}")
                check(not any(k.startswith(("log/", "artifact/")) for k in row),
                      "Stage tails leaked into a table")
            saved["tables"].append({"path": str(path), "records": records})
        check(any(len(t["records"]) == saved["summary"]["cases"]
                  for t in saved["tables"]), "Final cumulative table was not saved")
        stream = list(Path(saved["files_dir"]).parent.glob("run-*.wandb"))
        check(len(stream) == 1 and stream[0].stat().st_size > 0,
              "SDK did not save an offline event stream")
        saved["event_stream"] = str(stream[0])

    # add_file() must stage evidence before the logger deletes its temporary file.
    staged = []
    for directory in (output / "cache", output / "data"):
        for path in directory.rglob("*"):
            if not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            try:
                record = json.loads(path.read_bytes())
            except (ValueError, UnicodeError):
                continue
            if isinstance(record, dict) and {"config", "results", "profiles",
                                            "provenance_rules"} <= record.keys():
                check(record["config"] == cfg.to_dict(), "Staged config changed")
                check(record["provenance_rules"] == PROVENANCE_RULESET,
                      "Staged classifier ID changed")
                check(all(p in profiles for p in record["profiles"]),
                      "Staged profiles changed")
                staged.append(str(path))
    check(staged, "No staged raw-profile evidence found in offline SDK storage")
    report["staged_raw_evidence"] = staged
    report["status"] = "passed"
    destination = output / "report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Offline real-SDK smoke passed: {destination}")


if __name__ == "__main__":
    main()
