"""Opt-in W&B logging for rollouts and the H200 artifact backfill.

wandb is imported lazily inside functions only; this module stays importable
(and testable with fakes) without wandb, weave or torch installed.
"""

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# W&B's own variables; wandb.init reads them too. No entity → the API key's
# default entity. Set both in .env (see .env.example), never in code.
ENTITY = os.environ.get("WANDB_ENTITY") or None
PROJECT = os.environ.get("WANDB_PROJECT", "CUDA-gen")
MANIFEST = Path(__file__).with_name("h200_runs.json")


def weave_project(entity=None, project=None):
    """'entity/project' for weave.init, or just 'project' without an entity."""
    entity, project = entity or ENTITY, project or PROJECT
    return f"{entity}/{project}" if entity else project

_DROP_PREFIXES = ("log/", "artifact/")


def metrics_for_wandb(result):
    """Flat run_task metrics minus log/artifact tails."""
    return {k: v for k, v in result.items() if not k.startswith(_DROP_PREFIXES)}


def config_for_wandb(cfg_dict, environment=None, **extra):
    """cfg/* RolloutConfig + runtime/* preflight keys + flat extras."""
    config = {f"cfg/{k}": v for k, v in cfg_dict.items()}
    if environment:
        config.update({k: v for k, v in environment.items() if k.startswith("runtime/")})
    config.update(extra)
    return config


def _read(artifacts_dir, name, run_id):
    path = Path(artifacts_dir) / f"{name}-{run_id}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _case_row(case, r):
    return {"case": case, "backend": r.get("meta/backend"),
            "verify_ok": r.get("env/verify_ok"),
            "provenance": r.get("env/provenance"),
            "t_eager_us": r.get("env/t_eager_us"),
            "t_compile_us": r.get("env/t_compile_us"),
            "t_gen_us": r.get("env/t_gen_us"),
            "speedup_vs_compile": r.get("env/speedup_vs_compile"),
            "reward": r.get("reward/value"), "reward_base": r.get("reward/base"),
            "reason": r.get("reward/reason"),
            "overlong_penalty": r.get("reward/overlong_penalty"),
            "wall_s": r.get("env/wall_s"), "turns_used": r.get("env/turns_used")}


def _kernel_rows(kernel_names):
    rows = []
    for backend, rec in (kernel_names or {}).get("backends", {}).items():
        if not rec.get("measured"):
            rows.append({"backend": backend, "event": None, "us": None, "class": None,
                         "measured": False,
                         "reason": rec.get("reason") or rec.get("unavailable"),
                         "unmatched": ""})
            continue
        unmatched = ",".join(rec.get("unmatched_agent_names") or [])
        for event, k in rec.get("kernels", {}).items():
            rows.append({"backend": backend, "event": event, "us": k["us"],
                         "class": k["class"], "measured": True, "reason": None,
                         "unmatched": unmatched})
    return rows


def _gpu_tests_status(path):
    path = Path(path)
    if not path.is_file():
        return None
    matches = re.findall(r"^(OK|FAILED)", path.read_text(errors="replace"), re.M)
    return matches[-1] if matches else None


def _summary(rollouts, kernels, kernel_names, results, gpu_stderr):
    rewards = [r["reward"] for r in rollouts if r["reward"] is not None]
    return {"cases": len(rollouts),
            "verify_ok": sum(bool(r["verify_ok"]) for r in rollouts),
            "provenance_pass": sum(r["provenance"] is not None and r["provenance"] >= 0.9
                                   for r in rollouts),
            "reward_positive": sum(r["reward"] is not None and r["reward"] > 0
                                   for r in rollouts),
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "unmatched_agent_names_total": sum(
                len(rec.get("unmatched_agent_names") or [])
                for rec in (kernel_names or {}).get("backends", {}).values()),
            "backends_measured": sum(bool(rec.get("measured"))
                                     for rec in (kernel_names or {}).get("backends", {}).values()),
            "gpu_tests": _gpu_tests_status(gpu_stderr),
            "unavailable_backends": ",".join(sorted((results or {}).get("unavailable_backends") or {}))}


def backfill_payload(run_id, artifacts_dir, manifest=None):
    """Pure payload assembly for one manifest run; raises ValueError if unknown."""
    artifacts_dir = Path(artifacts_dir)
    manifest = manifest or json.loads(MANIFEST.read_text())
    entry = next((r for r in manifest["runs"] if r["run_id"] == run_id), None)
    if entry is None:
        raise ValueError(f"run_id not in manifest: {run_id}")
    results = _read(artifacts_dir, "results", run_id)
    kernel_names = _read(artifacts_dir, "kernel_names", run_id)
    environment = _read(artifacts_dir, "environment", run_id)
    versions = (results or {}).get("versions") or {}
    config = config_for_wandb({}, environment,
                              **{f"versions/{k}": v for k, v in versions.items()},
                              source_command=(results or {}).get("command"))
    rollout_map = (results or {}).get("rollouts", {})
    rollouts = [_case_row(c, r) for c, r in rollout_map.items()]
    kernels = _kernel_rows(kernel_names)
    files = [str(artifacts_dir / Path(f).name) for f in entry.get("files", [])
             if (artifacts_dir / Path(f).name).is_file()]
    json.dumps(files, allow_nan=False)
    return {"run_id": run_id, "name": run_id, "kind": entry["kind"],
            "status": entry["status"],
            "tags": [entry["kind"], entry["status"], "h200", "backfill"],
            "notes": entry.get("notes", ""), "config": config,
            "rollouts": rollouts, "kernels": kernels,
            "metrics_by_case": {c: metrics_for_wandb(r) for c, r in rollout_map.items()},
            "summary": _summary(rollouts, kernels, kernel_names, results,
                                artifacts_dir / f"gpu_tests.stderr-{run_id}.log"),
            "files": files}


def _table(wandb, rows, fallback_columns):
    columns = list(rows[0]) if rows else fallback_columns
    return wandb.Table(columns=columns,
                     data=[[row.get(c) for c in columns] for row in rows])


def log_backfill(payload, entity=ENTITY, project=PROJECT, wandb_module=None):
    """Log one backfill payload as a W&B run; returns the run url or id."""
    wandb = wandb_module or importlib.import_module("wandb")
    run = wandb.init(entity=entity, project=project,
                     id=f"h200-{payload['run_id']}".replace("/", "-")[:64],
                     name=payload["name"], job_type=payload["kind"],
                     tags=payload["tags"], notes=payload["notes"],
                     config=payload["config"], resume="allow")
    for i, (case, metrics) in enumerate(payload["metrics_by_case"].items()):
        run.log({**metrics, "case": case}, step=i)
    run.log({"rollouts": _table(wandb, payload["rollouts"], ["case"]),
             "kernels": _table(wandb, payload["kernels"], ["backend"])})
    run.summary.update(payload["summary"])
    artifact = wandb.Artifact(name=f"h200-{payload['run_id']}".replace("/", "-")[:64],
                              type="validation-run",
                              metadata={"kind": payload["kind"], "status": payload["status"]})
    for path in payload["files"]:
        artifact.add_file(path)
    run.log_artifact(artifact)
    run.finish(exit_code=0 if payload["status"] == "finished" else 1)
    return getattr(run, "url", run.id)


def log_rollout(result, cfg, run=None, wandb_module=None):
    """Log one run_task result's flat metrics to the given or a new run."""
    wandb = wandb_module or importlib.import_module("wandb")
    if run is None:
        run = wandb.init(entity=ENTITY, project=PROJECT,
                         config=config_for_wandb(cfg.to_dict()))
    run.log(metrics_for_wandb(result))
    return run


def main(argv=None):
    from utils.config import RolloutConfig
    from utils.rollout_env import run_task

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bf = sub.add_parser("backfill")
    bf.add_argument("--run-id", action="append", default=[])
    bf.add_argument("--all", action="store_true")
    bf.add_argument("--artifacts", default="artifacts")
    bf.add_argument("--dry-run", action="store_true")
    bf.add_argument("--no-weave", action="store_true")
    bf.add_argument("--entity", default=ENTITY)
    bf.add_argument("--project", default=PROJECT)
    ro = sub.add_parser("rollout")
    ro.add_argument("workdir")
    RolloutConfig.add_args(ro)
    ro.add_argument("--turns-used", type=int)
    ro.add_argument("--no-weave", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "backfill":
        manifest = json.loads(MANIFEST.read_text())
        run_ids = [r["run_id"] for r in manifest["runs"]] if args.all else args.run_id
        payloads = [backfill_payload(rid, args.artifacts, manifest) for rid in run_ids]
        if args.dry_run:
            print(json.dumps(payloads[0] if len(payloads) == 1 else payloads,
                             allow_nan=False))
            return
        if not args.no_weave:
            from tracking import weave_trace
            # Ops record only after weave.init; same entity/project as wandb.
            weave_trace.init(weave_project(args.entity, args.project))
        for payload in payloads:
            if not args.no_weave:
                weave_trace.replay_backfill(payload)
            print(log_backfill(payload, args.entity, args.project))
        return
    cfg = RolloutConfig.from_args(args)
    if args.no_weave:
        result = run_task(args.workdir, cfg, args.turns_used)
    else:
        from tracking import weave_trace
        weave_trace.init()
        result = weave_trace.traced_run_task(args.workdir, cfg, args.turns_used)
    log_rollout(result, cfg).finish()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])
