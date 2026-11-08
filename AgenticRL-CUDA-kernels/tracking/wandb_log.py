"""Opt-in W&B tables, aggregate summaries and iteration-level batch metrics.

No SDK or Torch imports until explicitly logging; dry-run payloads need no SDK.
See Architecture-decision/0002-wandb-tracking-contract.md.
"""
import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import weakref
from pathlib import Path

from utils.provenance import PROVENANCE_RULESET

ENTITY = os.environ.get("WANDB_ENTITY") or None
PROJECT = os.environ.get("WANDB_PROJECT", "CUDA-gen")
MANIFEST = Path(__file__).with_name("h200_runs.json")
SHARES = ("agent", "library", "torch", "memory")
BATCH_KEYS = ("adv/frac_dropped", "adv/mean", "adv/std", "reward/mean", "reward/frac_3",
              "reward/frac_provenance_fail", "reward/frac_max_turns",
              "reward/mean_overlong_penalty", "turns/mean", "turns/p95")


def weave_project(entity=None, project=None):
    entity, project = entity or ENTITY, project or PROJECT
    return f"{entity}/{project}" if entity else project


def metrics_for_wandb(result):
    """Only batch aggregate scalars belong in metric history, never case results."""
    return {k: v for k, v in result.items() if k in BATCH_KEYS}


def config_for_wandb(cfg_dict, environment=None, **extra):
    config = {f"cfg/{k}": v for k, v in cfg_dict.items()}
    config.update({k: v for k, v in (environment or {}).items() if k.startswith("runtime/")})
    config.update(extra)
    return config


def _read(artifacts_dir, name, run_id):
    path = Path(artifacts_dir) / f"{name}-{run_id}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _case_row(case, result, profile=None, candidate_sha256=None, rules=None, cfg=None):
    row = {k: v for k, v in result.items() if not k.startswith(("log/", "artifact/"))}
    row.update(case=case, **{"meta/model_new_sha256": candidate_sha256,
                            "meta/provenance_rules": rules,
                            "case_config": json.dumps(cfg, sort_keys=True) if cfg else None,
                            "timing_measured": (profile or {}).get("timing_measured")})
    # Never borrow shares from another fixture or from the separate name-recorder pass.
    for bucket in SHARES:
        row["env/provenance_" + bucket] = (profile or {}).get("provenance/" + bucket)
    return row


def _kernel_rows(kernel_names):
    rows = []
    for backend, rec in (kernel_names or {}).get("backends", {}).items():
        unmatched = ",".join(rec.get("unmatched_agent_names") or [])
        if not rec.get("measured") or not rec.get("kernels"):
            rows.append({"backend": backend, "event": None, "us": None, "class": None,
                         "measured": bool(rec.get("measured")),
                         "reason": rec.get("reason") or rec.get("unavailable"), "unmatched": unmatched})
        for event, k in rec.get("kernels", {}).items():
            rows.append({"backend": backend, "event": event, "us": k["us"], "class": k["class"],
                         "measured": True, "reason": None, "unmatched": unmatched})
    return rows


def _gpu_tests_status(path):
    path = Path(path)
    if not path.is_file():
        return None
    matches = re.findall(r"^(OK|FAILED)", path.read_text(errors="replace"), re.M)
    return matches[-1] if matches else None


def _summary(rollouts, kernel_names=None, results=None, gpu_stderr=None):
    rewards = [r["reward/value"] for r in rollouts if r.get("reward/value") is not None]
    thresholds = [(json.loads(r["case_config"]).get("provenance_threshold")
                   if r.get("case_config") else None) for r in rollouts]
    # Historical gate threshold is not recoverable from current defaults. The
    # milestone reason proves passage/failure, so counts use that recorded result.
    passed = sum((r.get("env/provenance") >= t if t is not None and r.get("env/provenance") is not None
                  else r.get("reward/reason") in ("correct", "faster_than_eager", "faster_than_both"))
                 for r, t in zip(rollouts, thresholds))
    return {"cases": len(rollouts), "verify_ok": sum(bool(r.get("env/verify_ok")) for r in rollouts),
            "provenance_pass": passed,
            "reward_positive": sum(r > 0 for r in rewards),
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "unmatched_agent_names_total": sum(len(r.get("unmatched_agent_names") or [])
                for r in (kernel_names or {}).get("backends", {}).values()),
            "backends_measured": sum(bool(r.get("measured")) for r in (kernel_names or {}).get("backends", {}).values()),
            "gpu_tests": _gpu_tests_status(gpu_stderr) if gpu_stderr else None,
            "unavailable_backends": ",".join(sorted((results or {}).get("unavailable_backends") or {}))}


def backfill_payload(run_id, artifacts_dir, manifest=None):
    artifacts_dir = Path(artifacts_dir)
    manifest = manifest or json.loads(MANIFEST.read_text())
    entry = next((r for r in manifest["runs"] if r["run_id"] == run_id), None)
    if entry is None:
        raise ValueError(f"run_id not in manifest: {run_id}")
    results = _read(artifacts_dir, "results", run_id) or {}
    names = _read(artifacts_dir, "kernel_names", run_id) or {}
    environment = _read(artifacts_dir, "environment", run_id)
    cfg = results.get("config") or {}
    # An old artifact's classifier is unknown; never label it with today's rules.
    rules = results.get("provenance_rules") or names.get("provenance_rules")
    config = config_for_wandb(cfg, environment,
        **{f"versions/{k}": v for k, v in results.get("versions", {}).items()},
        source_command=results.get("command"), **{"meta/provenance_rules": rules,
        "tracking/config_available": bool(cfg), "tracking/rules_available": rules is not None,
        "tracking/group_configs": results.get("group_configs"),
        "tracking/contract": "wandb-contract-v1"})
    rollout_map = results.get("rollouts", {})
    rows = [_case_row(c, r, results.get("profiles", {}).get(c),
        results.get("candidate_sha256", {}).get(c), rules, results.get("case_configs", {}).get(c))
        for c, r in rollout_map.items()]
    config["tracking/shares_available"] = any(r.get("env/provenance") is not None for r in rows) and all(
        all(r["env/provenance_" + b] is not None for b in SHARES)
        for r in rows if r.get("env/provenance") is not None)
    config["tracking/candidates_available"] = bool(rows) and all(r["meta/model_new_sha256"] for r in rows)
    metrics = {f"batch/{algo}/{k}": v for algo, group in results.get("measured_groups", {}).items()
               for k, v in metrics_for_wandb(group["stats"]).items()}
    return {"run_id": run_id, "name": run_id, "kind": entry["kind"], "status": entry["status"],
            "tags": [entry["kind"], entry["status"], "h200", "backfill"], "notes": entry.get("notes", ""),
            "config": config, "rollouts": rows, "kernels": _kernel_rows(names),
            "metrics_by_case": rollout_map,  # Weave replay only, including stage tails.
            "metric_keys": sorted(["iteration", *metrics]), "metrics": metrics,
            "summary": _summary(rows, names, results, artifacts_dir / f"gpu_tests.stderr-{run_id}.log"),
            "files": [str(artifacts_dir / Path(f).name) for f in entry.get("files", [])
                      if (artifacts_dir / Path(f).name).is_file()]}


def _table(wandb, rows, fallback_columns):
    columns = list(dict.fromkeys(k for row in rows for k in row)) if rows else fallback_columns
    return wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in rows])


def define_metrics(run):
    run.define_metric("iteration", summary="none")
    run.define_metric("*", step_metric="iteration", summary="none")


def _clear_case_summary(run):
    # A resumed old backfill may still contain last-case summary values.
    for key in list(run.summary.keys()):
        if key == "case" or key.startswith(("env/", "meta/", "bench/", "log/", "artifact/")) or key in (
                "reward/value", "reward/base", "reward/reason", "reward/overlong_penalty"):
            del run.summary[key]


def log_backfill(payload, entity=ENTITY, project=PROJECT, wandb_module=None, *, new_run_id=None):
    wandb = wandb_module or importlib.import_module("wandb")
    run = wandb.init(entity=entity, project=project,
        id=new_run_id or f"h200-{payload['run_id']}".replace("/", "-")[:64], name=payload["name"],
        job_type=payload["kind"], tags=payload["tags"], notes=payload["notes"],
        config=payload["config"], resume="never" if new_run_id else "allow")
    define_metrics(run)
    _clear_case_summary(run)
    # A validation invocation is one snapshot (iteration 0), not N training steps.
    run.log({"iteration": 0, **payload["metrics"],
             "rollouts": _table(wandb, payload["rollouts"], ["case"]),
             "kernels": _table(wandb, payload["kernels"], ["backend"])})
    run.summary.update(payload["summary"])
    artifact = wandb.Artifact(
        name=f"validation-{new_run_id}" if new_run_id
             else f"h200-{payload['run_id']}".replace("/", "-")[:64],
        type="validation-run", metadata={"kind": payload["kind"], "status": payload["status"]})
    for path in payload["files"]:
        artifact.add_file(path)
    run.log_artifact(artifact)
    run.finish(exit_code=0 if payload["status"] == "finished" else 1)
    return getattr(run, "url", run.id)


# Live-run rows and config, keyed by the run object. wandb's Run.__setattr__
# rejects unknown attributes, so this state cannot be stored on the run.
_LIVE = weakref.WeakKeyDictionary()


def _live_run(cfg, run, wandb, environment):
    if run is None:
        if environment is None:
            from utils.runtime import check
            environment = check()
        run = wandb.init(entity=ENTITY, project=PROJECT, name=cfg.run_name or None,
            group=cfg.algo, tags=cfg.tags, config=config_for_wandb(cfg.to_dict(), environment,
            **{"meta/provenance_rules": PROVENANCE_RULESET, "tracking/contract": "wandb-contract-v1"}))
    state = _LIVE.get(run)
    if state is None:
        define_metrics(run)
        state = _LIVE[run] = {"rows": [], "config": cfg.to_dict()}
    if state["config"] != cfg.to_dict():
        raise ValueError("Live runs require constant RolloutConfig; start a new run for changed config")
    return run


def _rows(run):
    return _LIVE[run]["rows"]


def _live_evidence(run, wandb, results, profiles, cfg, workdirs, step):
    """Attach raw records and candidate sources; stage tails stay out of metrics."""
    artifact = wandb.Artifact(name=f"rollout-{run.id}-{step}-{len(_rows(run))}",
                              type="rollout-evidence", metadata={"provenance_rules": PROVENANCE_RULESET})
    with tempfile.TemporaryDirectory(prefix="kernel-evidence-") as tmp:
        path = Path(tmp) / "records.json"
        path.write_text(json.dumps({"config": cfg.to_dict(), "results": results,
                                    "profiles": profiles, "provenance_rules": PROVENANCE_RULESET},
                                   allow_nan=False))
        artifact.add_file(str(path))
        for i, workdir in enumerate(workdirs or []):
            root = Path(workdir)
            paths = [root / n for n in ("model.py", "model_new.py", "binding.cpp", "binding_registry.h")]
            paths += sorted((root / "kernels").rglob("*"))
            for source in paths:
                if source.is_file() and source.suffix in (".py", ".cu", ".cpp", ".h"):
                    artifact.add_file(str(source), name=f"case-{i}/{source.relative_to(root)}")
        trusted = Path(__file__).resolve().parent.parent
        for source in sorted((trusted / "utils").glob("*.py")) + sorted(trusted.glob("requirements*.txt")):
            artifact.add_file(str(source), name=f"harness/{source.relative_to(trusted)}")
        run.log_artifact(artifact)


def log_rollout(result, cfg, run=None, wandb_module=None, *, profile=None, workdir=None,
                environment=None, step=0, candidate_sha256=None):
    """Append a case to a table; standalone evaluation has no batch advantages."""
    wandb = wandb_module or importlib.import_module("wandb")
    run = _live_run(cfg, run, wandb, environment)
    candidate = Path(workdir) / "model_new.py" if workdir else None
    sha = candidate_sha256 or (hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate and candidate.is_file() else None)
    row = _case_row(str(len(_rows(run))), result, profile, sha,
                    PROVENANCE_RULESET, cfg.to_dict())
    _rows(run).append(row)
    _clear_case_summary(run)
    run.log({"iteration": step, "rollouts": _table(wandb, _rows(run), ["case"])})
    run.summary.update(_summary(_rows(run)))
    _live_evidence(run, wandb, [result], [profile], cfg, [workdir] if workdir else None, step)
    return run


def log_batch(task_ids, results, cfg, *, step, run=None, profiles=None, workdirs=None,
              environment=None, wandb_module=None):
    """Live training hook: one same-algorithm run, one batch-stat point per iteration."""
    from utils.grpo import build_batch
    results = list(results)
    pairs, stats = build_batch(task_ids, results, cfg)
    wandb = wandb_module or importlib.import_module("wandb")
    run = _live_run(cfg, run, wandb, environment)
    for i, result in enumerate(results):
        # Accumulate rows without emitting per-rollout metric steps.
        candidate = Path(workdirs[i]) / "model_new.py" if workdirs else None
        sha = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate and candidate.is_file() else None
        row = _case_row(str(len(_rows(run))), result,
                        profiles[i] if profiles else None, sha, PROVENANCE_RULESET, cfg.to_dict())
        _rows(run).append(row)
    run.log({"iteration": step, **metrics_for_wandb(stats),
             "rollouts": _table(wandb, _rows(run), ["case"])})
    _clear_case_summary(run)
    run.summary.update(_summary(_rows(run)))
    _live_evidence(run, wandb, results, profiles, cfg, workdirs, step)
    return run, pairs, stats


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
    profile = {}
    candidate = Path(args.workdir) / "model_new.py"
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else None
    if args.no_weave:
        result = run_task(args.workdir, cfg, args.turns_used, profile_out=profile)
    else:
        from tracking import weave_trace
        weave_trace.init()
        result = weave_trace.traced_run_task(args.workdir, cfg, args.turns_used, profile_out=profile)
    log_rollout(result, cfg, profile=profile, workdir=args.workdir, candidate_sha256=candidate_sha256).finish()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])
