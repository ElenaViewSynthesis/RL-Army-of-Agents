"""Subprocess-isolated task evaluation; safe to import without Torch.

The deadline covers compile + verification + profiling together. Each child
gets a new POSIX process group, killed as a unit on timeout (including nvcc
and torch.compile workers). This provides crash/timeout isolation, not an OS
security sandbox for hostile code. Use a restricted container in phase 2.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from utils.backend import detect_backend
from utils.config import RolloutConfig
from utils.reward import compute_reward


def _stage(module, args, workdir, env, deadline):
    # Pin the trusted utils package before putting the task first for model.py,
    # kernels/, and cuda_extension.so. A task need not carry its own utils copy.
    harness = str(Path(__file__).resolve().parent.parent)
    script = (
        "import sys, runpy; "
        f"sys.path.insert(0, {harness!r}); import utils; "
        f"sys.path = [p for p in sys.path if p != {harness!r}]; "
        f"sys.path.insert(0, {str(workdir)!r}); "
        f"sys.argv = {[module, *args]!r}; "
        f"runpy.run_module({module!r}, run_name='__main__')"
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, "", "", True
    # Files prevent an agent's noisy output from consuming unbounded parent
    # RAM or keeping communicate() alive via a grandchild's inherited pipe.
    with tempfile.TemporaryFile(dir=workdir / ".cache") as stdout, tempfile.TemporaryFile(dir=workdir / ".cache") as stderr:
        proc = subprocess.Popen([sys.executable, "-c", script], cwd=workdir, env=env,
                                stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also clean up descendants if the direct child crashes/exits early.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                # SIGKILL may be pending in an uninterruptible driver syscall.
                # Never make the parent wait indefinitely for that kernel.
                timed_out = True
        def tail(stream, size):
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
        return proc.returncode, tail(stdout, 4 * 1024 * 1024), tail(stderr, 8192), timed_out


def run_task(workdir: str, cfg: RolloutConfig, turns_used: int | None = None) -> dict:
    """Return stable, flat, JSON-safe metrics; unmeasured GPU fields are None.

    CPU/interpreter runs can establish correctness, but return reward -1 with
    reason 'unmeasured'. Failure gates and timeouts never receive shaping.
    Different workdirs have independent build/JIT caches; GPU visibility is
    inherited verbatim. max_context_tokens is exposed for the future loop;
    this environment cannot enforce a policy's context length.
    """
    if turns_used is not None and (not isinstance(turns_used, int) or turns_used < 0):
        raise ValueError("turns_used must be a non-negative integer or None")
    start = time.monotonic()
    path = Path(workdir).expanduser().resolve()
    result = {
        "env/verify_ok": False,
        "env/t_eager_us": None,
        "env/t_compile_us": None,
        "env/t_gen_us": None,
        "env/speedup_vs_compile": None,
        "env/provenance": None,
        "env/wall_s": 0.0,
        "env/turns_used": turns_used,
        "reward/value": -1.0,
        "reward/base": -1,
        "reward/reason": "compile",
        "reward/overlong_penalty": 0.0,
        "meta/backend": cfg.backend,
        "meta/task_id": path.name,
        "meta/algo": cfg.algo,
        "meta/seed": cfg.seed,
        "log/compile_tail": "",
        "log/stderr_tail": "",
    }

    def finish(reason=None, reward=None):
        if reason is not None:
            result["reward/reason"] = reason
        if reward is not None:
            for target, source in (("value", "reward"), ("base", "reward_base"),
                                   ("reason", "reason"), ("overlong_penalty", "overlong_penalty")):
                result["reward/" + target] = reward[source]
        result["env/wall_s"] = time.monotonic() - start
        return result

    stage = "workdir"
    try:
        if not path.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {path}")
        stage = "backend"
        backend = detect_backend(path) if cfg.backend == "auto" else cfg.backend
        result["meta/backend"] = backend
        if turns_used is not None and turns_used > cfg.max_turns:
            return finish("max_turns")
        (path / ".cache").mkdir(exist_ok=True)
        env = os.environ.copy()
        # Do not touch CUDA_VISIBLE_DEVICES or caller process state.
        env.setdefault("TRITON_CACHE_DIR", str(path / ".cache" / "triton"))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(path / ".cache" / "inductor"))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        deadline = start + cfg.timeout_s
        stages = (("compile", "utils.compile", ["--backend", backend, "--seed", str(cfg.seed)]),
                  ("verification", "utils.verification", ["--seed", str(cfg.seed)]),
                  ("profiling", "utils.profiling", ["--backend", backend, "--iters", str(cfg.iters),
                                                    "--seed", str(cfg.seed), "--json"]))
        profile = None
        for stage, module, args in stages:
            code, stdout, stderr, timed_out = _stage(module, args, path, env, deadline)
            if stage == "compile":
                result["log/compile_tail"] = stdout[-8192:]
            if stderr:
                result["log/stderr_tail"] = (result["log/stderr_tail"] + f"[{stage}]\n" + stderr)[-8192:]
            if timed_out:
                return finish("timeout")
            if code != 0:
                result["log/stderr_tail"] = (result["log/stderr_tail"] + stdout)[-8192:]
                return finish(stage)
            if stage == "verification":
                if "[PASS] verify success" not in stdout:
                    return finish("verification")
                result["env/verify_ok"] = True
            if stage == "profiling":
                profile = json.loads(stdout.strip().splitlines()[-1])
        for field in ("t_eager_us", "t_compile_us", "t_gen_us", "provenance"):
            result["env/" + field] = profile[field]
        reward = compute_reward(result["env/verify_ok"], profile["t_eager_us"], profile["t_compile_us"],
                                profile["t_gen_us"], profile["provenance"], cfg, turns_used)
        if profile["t_gen_us"] is not None and profile["t_gen_us"] > 0:
            result["env/speedup_vs_compile"] = profile["t_compile_us"] / profile["t_gen_us"]
        # A malformed/non-finite measurement is an environment failure, never
        # an unserialisable metrics result.
        json.dumps(result, allow_nan=False)
        return finish(reward=reward)
    except (OSError, ValueError, TypeError, KeyError, IndexError, SyntaxError) as exc:
        result["log/stderr_tail"] = (result["log/stderr_tail"] + f"{type(exc).__name__}: {exc}")[-8192:]
        # Remove invalid measurements before exposing the failure to loggers.
        for field in ("t_eager_us", "t_compile_us", "t_gen_us", "provenance", "speedup_vs_compile"):
            result["env/" + field] = None
        return finish(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    RolloutConfig.add_args(parser)
    parser.add_argument("--turns-used", type=int)
    args = parser.parse_args()
    try:
        cfg = RolloutConfig.from_args(args)
        result = run_task(args.workdir, cfg, args.turns_used)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
