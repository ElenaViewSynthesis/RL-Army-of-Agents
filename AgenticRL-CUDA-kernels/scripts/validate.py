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
from utils.backend import available_backends
from utils.config import RolloutConfig
from utils.grpo import build_batch
from utils.reward import compute_reward
from utils.rollout_env import run_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-s", type=float, default=180)
    args = parser.parse_args()
    os.environ["CUDA_AGENT_DEVICE"] = args.device
    os.environ["TRITON_INTERPRET"] = "1" if args.device == "cpu" else "0"
    cfg = dataclasses.replace(RolloutConfig.preset("grpo"), timeout_s=args.timeout_s)
    results, unavailable = {}, {}
    with scratch("report-") as tmp:
        triton_task = make_task(tmp, "triton")
        for label, path, config, turns in (
            ("triton", triton_task, cfg, 1),
            ("hack", make_task(tmp, "hack"), cfg, 1),
            ("timeout", make_task(tmp, "sleep"), dataclasses.replace(cfg, timeout_s=5), 1),
            ("max_turns", triton_task, cfg, 25),
            ("overlong", triton_task, dataclasses.replace(RolloutConfig.preset("dapo"), timeout_s=args.timeout_s), 19),
        ):
            results[label] = run_task(str(path), config, turns)
            print(json.dumps({"case": label, **results[label]}, allow_nan=False), flush=True)
        if args.device == "cuda":
            results["cuda"] = run_task(str(make_task(tmp, "cuda")), cfg, 1)
            print(json.dumps({"case": "cuda", **results["cuda"]}, allow_nan=False), flush=True)
        # DSL backends have no CPU interpreter: run their honest and Torch-op hack
        # fixtures only on a GPU with the package installed; otherwise say why not.
        for name, missing in available_backends().items():
            if name in ("cuda", "triton"):
                continue
            reason = missing if args.device == "cuda" else "GPU only (no CPU interpreter)"
            if reason:
                unavailable[name] = reason
                print(json.dumps({"case": name, "unavailable": reason}), flush=True)
                continue
            for label in (name, f"hack-{name}"):
                results[label] = run_task(str(make_task(tmp, label)), cfg, 1)
                print(json.dumps({"case": label, **results[label]}, allow_nan=False), flush=True)
    # This demo tests reward/advantage arithmetic only, not GPU performance.
    synthetic = []
    for rid, provenance in (("triton", 1.0), ("cuda", 1.0), ("hack", 0.0)):
        reward = compute_reward(True, 100, 80, 50, provenance, cfg, 1)
        synthetic.append({"rollout_id": rid, "reward/value": reward["reward"],
                          "reward/reason": reward["reason"], "env/turns_used": 1})
    demo, measured = {}, {}
    for algo in ("grpo", "dapo"):
        config = dataclasses.replace(RolloutConfig.preset(algo), group_size=3)
        pairs, stats = build_batch(["axpby"] * 3, synthetic, config)
        demo[algo] = {"advantages": dict(pairs), "stats": stats}
        if all(k in results and results[k]["env/provenance"] is not None for k in ("triton", "cuda", "hack")):
            rows = [dict(results[k], rollout_id=k) for k in ("triton", "cuda", "hack")]
            pairs, stats = build_batch(["axpby"] * 3, rows, config)
            measured[algo] = {"advantages": dict(pairs), "stats": stats}
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for package in ("torch", "triton", "numpy", "tilelang", "cuda-tile", "nvidia-cutlass-dsl"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {"command": f"python -m scripts.validate --device {args.device} --timeout-s {args.timeout_s:g}",
              "device": args.device, "versions": versions, "rollouts": results,
              "unavailable_backends": unavailable,
              "measured_groups": measured, "synthetic_groups_not_gpu_measurements": demo}
    output = ROOT / "artifacts" / "results.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
