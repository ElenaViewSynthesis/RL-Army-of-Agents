"""Opt-in Weave tracing for rollouts and H200 backfill replay.

weave is imported lazily inside functions only; the module stays importable
without weave, wandb or torch installed.
"""

import hashlib
import importlib
from pathlib import Path

from tracking.wandb_log import weave_project
from utils.backend import detect_backend

_OPS = {}


def init(project=None):
    import weave
    return weave.init(project or weave_project())


def ops():
    """Weave ops created once, on first use."""
    if not _OPS:
        import weave
        from utils.config import RolloutConfig
        from utils.grpo import build_batch
        from utils.reward import compute_reward
        from utils.rollout_env import run_task

        def _run(inputs: dict) -> dict:
            # cfg stays a plain dict in the trace; traces keep the full
            # result, including log/* stage tails.
            return run_task(inputs["workdir"], RolloutConfig(**inputs["cfg"]),
                            inputs["turns_used"])

        _OPS.update(run_task=weave.op(name="run_task")(_run),
                    compute_reward=weave.op(name="compute_reward")(compute_reward),
                    build_batch=weave.op(name="build_batch")(build_batch))
    return _OPS


def traced_compute_reward(*args, **kwargs):
    return ops()["compute_reward"](*args, **kwargs)


def traced_build_batch(*args, **kwargs):
    return ops()["build_batch"](*args, **kwargs)


def rollout_inputs(workdir, cfg, turns_used):
    """The policy-produced model input: sources, hashes and config."""
    workdir = Path(workdir)
    model_new = workdir / "model_new.py"
    try:
        backend = detect_backend(workdir)
    except ValueError:
        backend = "invalid"
    kernels = workdir / "kernels"
    return {"task_id": workdir.name, "backend": backend, "cfg": cfg.to_dict(),
            "turns_used": turns_used,
            "model_new_sha256": hashlib.sha256(model_new.read_bytes()).hexdigest(),
            "model_new_source": model_new.read_text()[:20000],
            "kernel_files": sorted(str(p.relative_to(workdir))
                                   for p in kernels.rglob("*")
                                   if p.is_file() and "__pycache__" not in p.parts)}


def traced_run_task(workdir, cfg, turns_used=None):
    inputs = rollout_inputs(workdir, cfg, turns_used)
    inputs["workdir"] = str(workdir)
    return ops()["run_task"](inputs)


def replay_backfill(payload, weave_module=None):
    """Replay each backfilled rollout as one h200_rollout_replay Weave call."""
    weave = weave_module or importlib.import_module("weave")

    @weave.op(name="h200_rollout_replay")
    def _replay(run_id, case, backend, status):
        return {**payload["metrics_by_case"][case],
                "backfilled": True, "source": "artifacts"}

    for case, metrics in payload["metrics_by_case"].items():
        _replay(payload["run_id"], case, metrics.get("meta/backend"), payload["status"])
