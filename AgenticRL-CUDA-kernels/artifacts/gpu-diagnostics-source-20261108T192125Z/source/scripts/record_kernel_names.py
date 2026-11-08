"""Record profiler device kernel names per backend with the classifier's verdict.

For each available backend it stages compile + profiling on the honest fixture
(make_task) exactly like run_task, then maps every kernel_us/* event through
classify_kernel. Read artifacts/kernel_names.json before touching the
classifier. Torch-free in the parent process.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import utils.rollout_env
from tests.support import make_task, scratch
from utils.backend import BACKENDS, agent_kernel_names, available_backends
from utils.provenance import PROVENANCE_RULESET, classify_kernel
from utils.config import RolloutConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--output", default="artifacts/kernel_names.json")
    parser.add_argument("--backends", type=str, help="Comma list; default all of BACKENDS")
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    return args


def record_backend(name, timeout_s):
    """compile + profiling on the honest fixture; stage failures become data."""
    cfg = RolloutConfig(backend=name, timeout_s=timeout_s)
    with scratch("kernels-") as tmp:
        task = make_task(tmp, name)
        (task / ".cache").mkdir(exist_ok=True)
        env = os.environ.copy()
        for spec in BACKENDS.values():
            for var, subdir in spec.cache_env:
                env.setdefault(var, str(task / ".cache" / subdir))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(task / ".cache" / "inductor"))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        deadline = time.monotonic() + timeout_s
        code, _, stderr, timed_out = utils.rollout_env._stage(
            "utils.compile", ["--backend", cfg.backend, "--seed", str(cfg.seed)], task, env, deadline)
        if timed_out:
            return {"measured": False, "reason": "timeout"}
        if code != 0:
            return {"measured": False, "reason": "compile failed", "stderr_tail": stderr}
        code, stdout, stderr, timed_out = utils.rollout_env._stage(
            "utils.profiling",
            ["--backend", cfg.backend, "--iters", str(cfg.iters), "--seed", str(cfg.seed), "--json"],
            task, env, deadline)
        if timed_out:
            return {"measured": False, "reason": "timeout"}
        if code != 0:
            return {"measured": False, "reason": "profiling failed", "stderr_tail": stderr}
        try:
            profile = json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"measured": False, "reason": "malformed profile output"}
        if not profile.get("timing_measured"):
            return {"measured": False, "reason": profile.get("timing_reason") or "CPU (no device events)",
                    **({"profile": profile} if profile.get("timing_reason") else {})}
        names = agent_kernel_names(task, name)
        kernels = {key[len("kernel_us/"):]: {"us": us,
                                            "class": classify_kernel(key[len("kernel_us/"):], names, name)}
                   for key, us in profile.items() if key.startswith("kernel_us/")}
        return {"measured": True, "agent_kernel_names": names, "profile": profile,
                "t_gen_us": profile["t_gen_us"], "provenance": profile["provenance"],
                **{f"provenance/{k}": profile[f"provenance/{k}"]
                   for k in ("agent", "library", "torch", "memory")},
                "kernels": kernels,
                "unmatched_agent_names": [n for n in names
                                          if not any(classify_kernel(e, [n], name) == "agent" for e in kernels)]}


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_AGENT_DEVICE"] = args.device
    os.environ["TRITON_INTERPRET"] = "1" if args.device == "cpu" else "0"
    names = ([x.strip() for x in args.backends.split(",") if x.strip()]
             if args.backends else list(BACKENDS))
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backends: {unknown}")
    available = available_backends()
    results = {}
    for name in names:
        results[name] = ({"unavailable": available[name]} if available.get(name)
                         else record_backend(name, args.timeout_s))
        print(json.dumps({"backend": name, **results[name]}, allow_nan=False), flush=True)
    payload = {"command": f"python -m scripts.record_kernel_names --device {args.device}"
                          f" --timeout-s {args.timeout_s:g}"
                          + (f" --backends {args.backends}" if args.backends else ""),
               "config": RolloutConfig(timeout_s=args.timeout_s).to_dict(),
               "case_configs": {n: RolloutConfig(backend=n, timeout_s=args.timeout_s).to_dict() for n in names},
               "config_scope": "compile and profiling only; reward and verification not evaluated",
               "provenance_rules": PROVENANCE_RULESET,
               "device": args.device, "backends": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return payload


if __name__ == "__main__":
    main(sys.argv[1:])
