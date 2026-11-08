import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from tests.support import make_task, scratch
from utils.backend import BACKENDS
from utils.profile_diagnostic import ALL_PREPARATIONS, ORDERS, PREPARATIONS
from utils.rollout_env import _stage


def plan(backends, repeat, preparations=PREPARATIONS):
    return [(backend, index, preparation, order)
            for index in range(repeat)
            for backend in backends
            for preparation in (preparations if index % 2 == 0 else preparations[::-1])
            for order in (ORDERS if index % 2 == 0 else ORDERS[::-1])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="triton,cutedsl")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--preparations", default=",".join(PREPARATIONS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    backends = args.backends.split(",")
    preparations = tuple(args.preparations.split(","))
    if (any(b not in BACKENDS for b in backends) or args.repeat <= 0
            or any(p not in ALL_PREPARATIONS for p in preparations)
            or len(set(preparations)) != len(preparations)):
        parser.error("known backends, distinct known preparations and positive repeat required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for trial, (backend, index, preparation, order) in enumerate(plan(backends, args.repeat, preparations)):
        target = output / f"{trial:03d}-{backend}-{preparation}"
        target.mkdir()
        row = {"trial": trial, "backend": backend, "repeat": index,
               "preparation": preparation, "order": order, "stages": {},
               "diagnostic_only": True}
        with scratch("order-") as tmp:
            task = make_task(tmp, backend)
            (task / ".cache").mkdir(exist_ok=True)
            candidate = (task / "model_new.py").read_bytes()
            row["candidate_sha256"] = hashlib.sha256(candidate).hexdigest()
            (target / "model_new.py").write_bytes(candidate)
            env = dict(os.environ, CUDA_AGENT_DEVICE="cuda", TRITON_INTERPRET="0",
                       PYTHONDONTWRITEBYTECODE="1")
            for spec in BACKENDS.values():
                for variable, subdir in spec.cache_env:
                    env[variable] = str(task / ".cache" / subdir)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(task / ".cache" / "inductor")
            stages = ([("compile", "utils.compile", ["--backend", backend])]
                      if backend == "cuda" else [])
            stages.append(("profiling", "utils.profile_diagnostic",
                           ["--backend", backend, "--order", order, "--preparation", preparation]))
            for stage, module, argv in stages:
                start = time.monotonic()
                code, stdout, stderr, timed_out = _stage(module, argv, task, env, start + 600)
                row["stages"][stage] = {"returncode": code, "timed_out": timed_out,
                                        "wall_s": time.monotonic() - start}
                (target / f"{stage}.stdout.log").write_text(stdout)
                (target / f"{stage}.stderr.log").write_text(stderr)
                if code != 0 or timed_out:
                    row["reason"] = f"{stage}_failed"
                    break
                if stage == "profiling":
                    try:
                        row.update(json.loads(stdout.strip().splitlines()[-1]))
                    except (ValueError, IndexError):
                        row["reason"] = "malformed_profile"
        (target / "record.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("trial", "backend", "preparation", "order")},
                         allow_nan=False), flush=True)
    (output / "records.json").write_text(json.dumps(
        {"diagnostic_only": True, "rows": rows}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
