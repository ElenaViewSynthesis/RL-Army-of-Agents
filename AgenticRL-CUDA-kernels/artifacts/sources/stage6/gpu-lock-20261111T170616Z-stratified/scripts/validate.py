"""Generate real rollout evidence and a separately labelled synthetic group demo.

Run from this project root: python -m scripts.validate [--device cpu|cuda].
All fixtures, caches and output stay inside this project. Stock model_new.py
is never replaced. Timings/provenance are null when no GPU is available.
"""

import argparse
import dataclasses
import json
import os

from tests.support import ROOT, make_task, scratch
from utils.backend import BACKENDS, available_backends
from utils.config import ALGOS, RolloutConfig
from utils.grpo import build_batch
from utils.provenance import PROVENANCE_RULESET
from utils.reward import compute_reward
from utils.rollout_env import run_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--backends", help="Comma-separated selection; default all backends")
    parser.add_argument("--suite", help="Progressive task names, all, bootstrap, or stage1..stage6")
    parser.add_argument("--case-set", default="smoke", help="smoke, stratified (Stage 6), all, acceptance, or one exact case id")
    parser.add_argument("--fixtures", default="honest,hack,wrong")
    parser.add_argument("--purpose", choices=("eval", "train"), default="eval")
    parser.add_argument("--suite-out", help="New output directory; existing evidence is never overwritten")
    parser.add_argument("--benchmark", action="store_true", help="Opt-in diagnostic benchmark for the suite")
    parser.add_argument("--cuda-graphs", action="store_true", help="Diagnostic graph capture and replay; requires --benchmark")
    parser.add_argument("--cold-start", action="store_true", help="Separate first-forward diagnostic sample")
    args = parser.parse_args()
    if args.cuda_graphs and not args.benchmark:
        parser.error("--cuda-graphs requires --benchmark")
    if args.cuda_graphs and not args.suite:
        parser.error("--cuda-graphs applies only to --suite runs")
    if args.suite:
        from datetime import datetime, timezone
        from uuid import uuid4
        from utils.task_suite import run_suite
        output = args.suite_out or str(ROOT / "artifacts" / (
            "task-suite-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]))
        try:
            report = run_suite(args.suite.split(","), output, device=args.device,
                               case_set=args.case_set, purpose=args.purpose,
                               backends=args.backends.split(",") if args.backends else None,
                               variants=args.fixtures.split(","), timeout_s=args.timeout_s,
                               benchmark=args.benchmark, cold_start=args.cold_start, cuda_graphs=args.cuda_graphs)
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        print(json.dumps({"suite/status": report["status"], "suite/output": output,
                          **{"suite/" + key: val for key, val in report["counts"].items()}}))
        raise SystemExit(1 if report["status"] == "fail" else 0)
    selected = set(args.backends.split(",")) if args.backends else set(BACKENDS)
    if not selected or selected - set(BACKENDS):
        parser.error("unknown or empty backend selection")
    os.environ["CUDA_AGENT_DEVICE"] = args.device
    os.environ["TRITON_INTERPRET"] = "1" if args.device == "cpu" else "0"
    cfg = dataclasses.replace(RolloutConfig.preset("grpo"), timeout_s=args.timeout_s)
    results, unavailable = {}, {}
    case_configs, profiles, candidates, group_configs = {}, {}, {}, {}
    stage_wall_s = {}

    def evaluate(label, path, config, turns):
        import hashlib
        case_configs[label] = config.to_dict()
        candidates[label] = hashlib.sha256((path / "model_new.py").read_bytes()).hexdigest()
        profile = {}
        kwargs = {"profile_out": profile}
        if os.environ.get("CUDA_AGENT_STAGE_TIMINGS") == "1":
            stage_wall_s[label] = {}
            kwargs["stage_times_out"] = stage_wall_s[label]
        result = run_task(str(path), config, turns, **kwargs)
        profiles[label] = profile
        return result

    with scratch("report-") as tmp:
        triton_task = make_task(tmp, "triton")
        for label, path, config, turns in (
            ("triton", triton_task, cfg, 1),
            ("reward-hack-torchop", make_task(tmp, "reward-hack-torchop"), cfg, 1),
            ("timeout", make_task(tmp, "sleep"), dataclasses.replace(cfg, timeout_s=5), 1),
            ("max_turns", triton_task, cfg, 25),
            ("overlong", triton_task, dataclasses.replace(RolloutConfig.preset("dapo"), timeout_s=args.timeout_s), 19),
        ):
            if "triton" not in selected:
                continue
            results[label] = evaluate(label, path, config, turns)
            print(json.dumps({"case": label, **results[label]}, allow_nan=False), flush=True)
        if args.device == "cuda" and "cuda" in selected:
            results["cuda"] = evaluate("cuda", make_task(tmp, "cuda"), cfg, 1)
            print(json.dumps({"case": "cuda", **results["cuda"]}, allow_nan=False), flush=True)
            label = "reward-hack-torchop-cuda"
            results[label] = evaluate(label, make_task(tmp, label), cfg, 1)
            print(json.dumps({"case": label, **results[label]}, allow_nan=False), flush=True)
        # DSL backends have no CPU interpreter: run their honest and Torch-op hack
        # fixtures only on a GPU with the package installed; otherwise say why not.
        for name, missing in available_backends().items():
            if name in ("cuda", "triton") or name not in selected:
                continue
            reason = missing if args.device == "cuda" else "GPU only (no CPU interpreter)"
            if reason:
                unavailable[name] = reason
                print(json.dumps({"case": name, "unavailable": reason}), flush=True)
                continue
            for label in (name, f"reward-hack-torchop-{name}"):
                results[label] = evaluate(label, make_task(tmp, label), cfg, 1)
                print(json.dumps({"case": label, **results[label]}, allow_nan=False), flush=True)
    # This demo tests reward/advantage arithmetic only, not GPU performance.
    synthetic = []
    for rid, provenance in (("triton", 1.0), ("cuda", 1.0), ("reward-hack-torchop", 0.0)):
        reward = compute_reward(True, 100, 80, 50, provenance, cfg, 1)
        synthetic.append({"rollout_id": rid, "reward/value": reward["reward"],
                          "reward/reason": reward["reason"], "env/turns_used": 1})
    demo, measured = {}, {}
    for algo in ALGOS:
        config = dataclasses.replace(RolloutConfig.preset(algo), group_size=3)
        group_configs[algo] = config.to_dict()
        pairs, stats = build_batch(["axpby"] * 3, synthetic, config)
        demo[algo] = {"advantages": dict(pairs), "stats": stats}
        if all(k in results and results[k]["env/provenance"] is not None for k in ("triton", "cuda", "reward-hack-torchop")):
            rows = [dict(results[k], rollout_id=k) for k in ("triton", "cuda", "reward-hack-torchop")]
            pairs, stats = build_batch(["axpby"] * 3, rows, config)
            measured[algo] = {"advantages": dict(pairs), "stats": stats}
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for package in ("torch", "triton", "numpy", "tilelang", "cuda-tile", "nvidia-cutlass-dsl"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {"command": f"python -m scripts.validate --device {args.device} --timeout-s {args.timeout_s:g}"
                         + (f" --backends {args.backends}" if args.backends else ""),
              "config": cfg.to_dict(), "case_configs": case_configs,
              "group_configs": group_configs, "profiles": profiles,
              "stage_wall_s": stage_wall_s,
              "candidate_sha256": candidates, "provenance_rules": PROVENANCE_RULESET,
              "device": args.device, "versions": versions, "rollouts": results,
              "unavailable_backends": unavailable,
              "measured_groups": measured, "synthetic_groups_not_gpu_measurements": demo}
    output = ROOT / "artifacts" / "results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
