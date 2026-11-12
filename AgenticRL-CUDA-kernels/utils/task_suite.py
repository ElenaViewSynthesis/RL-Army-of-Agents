"""Progressive workload evaluation with an explicit task-only provenance policy."""

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

from utils.backend import BACKENDS, available_backends
from utils.config import RolloutConfig
from utils.gpu_lock import MeasurementLock
from utils.provenance import PROVENANCE_RULESET
from utils.rollout_env import _stage, run_task
from utils.task_catalog import SOURCES, TASKS, get_case, get_task, select_cases, validate_usage, workload
from utils.task_policy import apply_task_policy, diagnostic_rates, gemm_ceiling_issue


ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("honest", "hack", "wrong")
EXTRA_VARIANTS = ("wrong_ancestor", "chain_replay", "wrong_padding", "wrong_skip")


@contextlib.contextmanager
def case_environment(task_name, case_id, device, cold_start=False, sample_dir=None):
    values = {"CUDA_AGENT_TASK": task_name, "CUDA_AGENT_TASK_CASE": case_id,
              "CUDA_AGENT_DEVICE": device, "TRITON_INTERPRET": "0" if device == "cuda" else "1",
              "CUDA_AGENT_BENCH_COLD": "1" if cold_start else "0"}
    if sample_dir is not None:
        values["CUDA_AGENT_BENCH_SAMPLES_DIR"] = str(sample_dir)
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def cuda_unavailable(device):
    if device != "cuda":
        return "CPU evaluation: GPU measurements and GPU candidate acceptance unavailable"
    try:
        import torch
    except ImportError:
        return "torch unavailable"
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    return None


def scaffold(task_name, backend, variant, destination):
    get_task(task_name)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend}")
    if variant not in (*VARIANTS, *EXTRA_VARIANTS):
        raise ValueError(f"Unknown fixture variant {variant}")
    if variant == "chain_replay" and (task_name != "eagle_gdn_tree_recurrence" or backend != "triton"):
        raise FileNotFoundError("chain_replay is the GDN Triton alternate solution")
    source = ROOT / "tasks" / task_name
    candidate = source / (f"model_new_{backend}.py" if variant == "honest" else f"model_new_{variant}.py")
    if not candidate.is_file():
        raise FileNotFoundError(f"{task_name}/{candidate.name}: fixture not implemented")
    destination = Path(destination)
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns(".cache", ".artifacts", "build", "__pycache__", "*.so"))
    for name in ("binding.cpp", "binding_registry.h"):
        if not (destination / name).is_file():
            shutil.copy2(ROOT / name, destination / name)
    text = candidate.read_text()
    if variant not in ("honest", "chain_replay"):
        if text.count('BACKEND = "triton"') != 1:
            raise ValueError("Negative fixture must contain one trusted backend placeholder")
        text = text.replace('BACKEND = "triton"', f"BACKEND = {backend!r}", 1)
    candidate_path = destination / "model_new.py"
    candidate_path.chmod(candidate_path.stat().st_mode | stat.S_IWUSR)
    candidate_path.write_text(text)
    return destination


def expected_outcome(result, variant):
    if variant in ("honest", "chain_replay"):
        return (result.get("env/verify_ok") is True and result.get("reward/base", -1) > 0
                and result.get("task/policy_violation") is None)
    if variant == "hack":
        return (result.get("env/verify_ok") is True and result.get("reward/value") == -1
                and result.get("reward/reason") == "provenance")
    return (result.get("env/verify_ok") is False and result.get("reward/value") == -1
            and result.get("reward/reason") == "verification")


def candidate_sources(path):
    files = [*path.glob("model_new*.py"), *(path / "kernels").rglob("*")]
    return {str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted(files)
            if item.is_file() and item.suffix in (".py", ".cu", ".cpp", ".h")
            and not any(part in (".cache", "__pycache__") for part in item.relative_to(path).parts)}


NOT_APPLICABLE_PREFIX = "control_not_applicable_"


def evaluate_case(task_name, case_id, backend, variant, destination, *, device="cuda",
                  purpose="eval", cfg=None, cold_start=False):
    started = time.monotonic()
    spec, item = get_task(task_name), get_case(task_name, case_id)
    if variant not in (*VARIANTS, *EXTRA_VARIANTS):
        raise ValueError(f"Unknown fixture variant {variant}")
    result = {"suite/task": task_name, "suite/case": case_id, "suite/stage": spec.stage,
              "suite/backend": backend, "suite/fixture": variant, "suite/purpose": purpose,
              "suite/eval_only": spec.eval_only, "suite/contract_only": spec.contract_only,
              "suite/status": "unavailable", "suite/reason": None}
    validate_usage(task_name, purpose)
    if spec.contract_only:
        result["suite/reason"] = "contract_only"
        return result
    if backend not in spec.backends:
        result["suite/reason"] = "backend_fixture_not_provided"
        return result
    if task_name.startswith("lora_") and variant in ("wrong_skip", "wrong_padding"):
        from utils.lora_contracts import token_slots, slot_ranks
        if variant == "wrong_skip" and -1 not in token_slots(item.params):
            result["suite/reason"] = "control_not_applicable_no_skipped_tokens"
            return result
        padded = (item.params["rank"]+15)//16*16
        if variant == "wrong_padding" and min(slot_ranks(item.params)) == padded:
            result["suite/reason"] = "control_not_applicable_no_rank_padding"
            return result
    reason = cuda_unavailable(device) or available_backends()[backend]
    if reason is None and backend == "cuda" and shutil.which("nvcc") is None:
        reason = "nvcc unavailable"
    if reason is not None:
        result["suite/reason"] = reason
        return result
    try:
        path = scaffold(task_name, backend, variant, destination)
    except FileNotFoundError as exc:
        result["suite/reason"] = str(exc)
        return result
    requested = cfg or RolloutConfig()
    # Only honest solutions are benchmarked. Graph replay is part of the
    # benchmark, so a fixture that is not benchmarked must drop cuda_graphs too
    # (RolloutConfig rejects cuda_graphs without benchmark).
    benchmarked = requested.benchmark and variant in ("honest", "chain_replay")
    config = dataclasses.replace(
        requested, backend=backend, benchmark=benchmarked,
        cuda_graphs=requested.cuda_graphs and benchmarked,
        exact_integer_outputs=True,
        relative_error_bounds=spec.stage == 6 and task_name != "lora_segment_metadata",
        provenance_threshold=max(requested.provenance_threshold,
                                 1.0 if task_name == "gemm_tensor_core" else .9))
    deadline = started + config.timeout_s
    protected = ("model.py", "binding.cpp", "binding_registry.h")
    before = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in protected}
    candidate_before = hashlib.sha256((path / "model_new.py").read_bytes()).hexdigest()
    sources_before = candidate_sources(path)
    profile = {}
    with case_environment(task_name, case_id, device, cold_start, path / "benchmark-samples"):
        raw = run_task(str(path), dataclasses.replace(
            config, timeout_s=max(.001, deadline - time.monotonic())), turns_used=1, profile_out=profile)
    deadline += raw.get("meta/gpu_lock_wait_s", 0.0)
    graded = apply_task_policy(raw, profile, config, turns_used=1)
    changed = any(not (path / name).is_file()
                  or hashlib.sha256((path / name).read_bytes()).hexdigest() != digest
                  for name, digest in before.items())
    result.update(graded)
    if changed:
        result.update({"suite/status": "failed", "suite/reason": "protected_task_files_modified",
                       "reward/value": -1.0, "reward/base": -1, "reward/reason": "verification",
                       "reward/overlong_penalty": 0.0})
    elif result.get("reward/reason") == "gpu_busy":
        result.update({"suite/status": "unavailable", "suite/reason": "gpu_busy"})
    else:
        passed = expected_outcome(result, variant)
        result.update({"suite/status": "passed" if passed else "failed",
                       "suite/reason": "expected_outcome" if passed else result.get("reward/reason")})
    perf_required = (task_name == "gemm_tensor_core" and variant == "honest"
                     and backend == "triton" and item.dtype == "bfloat16"
                     and item.params == {"m": 4096, "n": 4096, "k": 4096})
    result.update({"task/performance_required": perf_required,
                   "task/minimum_cublas_fraction": 0.6 if perf_required else None,
                   "task/cublas_fraction": None, "task/performance_accepted": None})
    if task_name == "gemm_tensor_core" and config.benchmark and result["suite/status"] == "passed":
        env = os.environ.copy()
        for spec_backend in BACKENDS.values():
            for variable, subdir in spec_backend.cache_env:
                env.setdefault(variable, str(path / ".cache" / subdir))
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(path / ".cache" / "inductor"))
        samples = path / "ceiling-samples.json"
        with case_environment(task_name, case_id, device):
            env.update({key: os.environ[key] for key in
                        ("CUDA_AGENT_TASK", "CUDA_AGENT_TASK_CASE", "CUDA_AGENT_DEVICE", "TRITON_INTERPRET")})
            # This auxiliary timing stage runs after run_task releases its lock.
            # Reacquire the same device lock; numerical-only LoRA checks below
            # are verification work and do not take a measurement lock.
            with MeasurementLock(env, config.gpu_lock, config.gpu_lock_timeout_s) as lock:
                wait = lock.metadata["meta/gpu_lock_wait_s"]
                previous_wait = result.get("meta/gpu_lock_wait_s", 0.0)
                result.update(lock.metadata)
                result["meta/gpu_lock_wait_s"] = previous_wait + wait
                deadline += wait
                if lock.busy:
                    result.update({"suite/status": "unavailable", "suite/reason": "gpu_busy",
                                   "reward/value": -1.0, "reward/base": -1,
                                   "reward/reason": "gpu_busy", "reward/overlong_penalty": 0.0})
                    code, stdout, stderr, timed_out = None, "", "", False
                else:
                    code, stdout, stderr, timed_out = _stage(
                        "utils.task_ceiling", ["--seed", str(config.seed), "--samples-json", str(samples)],
                        path, env, deadline)
        (path / "ceiling.stdout.log").write_text(stdout)
        (path / "ceiling.stderr.log").write_text(stderr)
        try:
            parsed = json.loads(stdout.strip().splitlines()[-1]) if not timed_out and code == 0 else {}
            ceiling = {key: value for key, value in parsed.items() if key.startswith("ceiling/")} \
                if isinstance(parsed, dict) else {}
        except (ValueError, IndexError):
            ceiling = {}
        issue = gemm_ceiling_issue(ceiling, item.params, item.dtype)
        result["ceiling/available"] = issue is None
        if result["ceiling/available"]:
            result.update(ceiling)
            fraction = ceiling.get("ceiling/cublas_fraction")
            result["task/cublas_fraction"] = fraction
            if perf_required:
                if ("H200" not in str(ceiling.get("ceiling/device_name", ""))
                        or ceiling.get("ceiling/device_capability") != "9.0"):
                    result["ceiling/available"] = False
                    result["ceiling/unavailable_reason"] = "fixture acceptance requires an H200 measurement"
                else:
                    result["task/performance_accepted"] = fraction >= .6
                if result["task/performance_accepted"] is False:
                    result.update({"suite/status": "failed", "suite/reason": "below_60_percent_cublas"})
        else:
            result["ceiling/unavailable_reason"] = (
                "case_deadline" if timed_out else f"ceiling_stage_exit_{code}" if code != 0 else issue)
    if perf_required and result["suite/status"] == "passed" and result["task/performance_accepted"] is None:
        result.update({"suite/status": "unavailable",
                       "suite/reason": "cuBLAS ceiling not yet recorded"})
    numerics = None
    if task_name.startswith("lora_") and task_name != "lora_segment_metadata" and variant == "honest" and result.get("env/verify_ok") and result.get("reward/reason") != "gpu_busy":
        target = path / "lora-numerics.json"
        env = os.environ.copy()
        with case_environment(task_name, case_id, device):
            env.update({key: os.environ[key] for key in
                        ("CUDA_AGENT_TASK", "CUDA_AGENT_TASK_CASE", "CUDA_AGENT_DEVICE", "TRITON_INTERPRET")})
            code, stdout, stderr, timed_out = _stage(
                "utils.lora_diagnostics", ["--output", str(target)], path, env, deadline)
        (path / "lora-numerics.stdout.log").write_text(stdout)
        (path / "lora-numerics.stderr.log").write_text(stderr)
        result["artifact/lora_numerics"] = str(target) if code == 0 and target.is_file() else None
        result["suite/numerics_unavailable_reason"] = (None if result["artifact/lora_numerics"] else
            "case_deadline" if timed_out else f"diagnostic_stage_exit_{code}")
        if result["artifact/lora_numerics"]:
            try:
                numerics = json.loads(target.read_text())
            except (OSError, ValueError):
                result["artifact/lora_numerics"] = None
                result["suite/numerics_unavailable_reason"] = "diagnostic_invalid_report"
    after = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
             if (path / name).is_file() else None for name in protected}
    if after != before:
        result.update({"suite/status": "failed", "suite/reason": "protected_task_files_modified",
                       "reward/value": -1.0, "reward/base": -1, "reward/reason": "verification",
                       "reward/overlong_penalty": 0.0, "ceiling/available": False,
                       "ceiling/unavailable_reason": "protected_task_files_modified",
                       "task/cublas_fraction": None, "task/performance_accepted": None})
    latency = result.get("bench/generated/event_us_median") if result.get("bench/available") else None
    result.update(diagnostic_rates(workload(task_name, item), latency))
    result["suite/wall_s"] = time.monotonic() - started
    bounds_path = path / ".cache" / "relative-error-bounds.json"
    bounds = None
    if config.relative_error_bounds and bounds_path.is_file():
        try:
            bounds = json.loads(bounds_path.read_text())
            result["artifact/relative_error_bounds"] = str(bounds_path)
        except (OSError, ValueError):
            pass  # Artifact content never determines a reward.
    payload = {"schema": "progressive-task-case-v1", "case": item.to_dict(),
               "config": config.to_dict(), "protected_sha256": before, "protected_sha256_after": after,
               "provenance_rules": PROVENANCE_RULESET,
               "source_provenance": [SOURCES[key] for key in spec.sources],
               "candidate_sha256": candidate_before,
               "candidate_sha256_after": hashlib.sha256((path / "model_new.py").read_bytes()).hexdigest(),
               "candidate_source_sha256": sources_before,
               "candidate_source_sha256_after": candidate_sources(path),
               "profile": profile, "result": result,
               **({"verification_bounds": bounds} if bounds is not None else {}), **({"lora_numerics": numerics} if numerics else {})}
    (path / "suite-case.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return result


def select_tasks(names):
    result = []
    for name in names:
        if name == "all":
            result.extend(TASKS)
        elif name in ("bootstrap", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"):
            stage = 0 if name == "bootstrap" else int(name[-1])
            result.extend(task.name for task in TASKS.values() if task.stage == stage)
        else:
            result.append(get_task(name).name)
    return list(dict.fromkeys(result))


def run_suite(names, output, *, case_set="smoke", backends=None, variants=VARIANTS,
              device="cuda", purpose="eval", timeout_s=600, benchmark=False, cold_start=False, cuda_graphs=False):
    names = select_tasks(names)
    if not names or not variants or set(variants) - set((*VARIANTS, *EXTRA_VARIANTS, "all")):
        raise ValueError("Task and fixture selections must be nonempty and valid")
    if backends is not None and (not backends or set(backends) - set(BACKENDS)):
        raise ValueError("Backend selections must be nonempty and valid")
    if cuda_graphs and not benchmark:
        raise ValueError("cuda_graphs requires the diagnostic benchmark")
    if cold_start and not benchmark:
        raise ValueError("cold_start requires the diagnostic benchmark")
    for name in names:
        get_task(name)
        validate_usage(name, purpose)
        select_cases(name, case_set)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    records = []
    config = dataclasses.replace(RolloutConfig(), timeout_s=timeout_s, benchmark=benchmark, cuda_graphs=cuda_graphs)
    for name in names:
        spec = get_task(name)
        chosen = tuple(backends) if backends else spec.backends or ("triton",)
        for item in select_cases(name, case_set):
            for backend in chosen:
                selected_variants = variants
                if "all" in variants:
                    selected_variants = (*VARIANTS,
                        *(("wrong_ancestor",) if (ROOT / "tasks" / name / "model_new_wrong_ancestor.py").is_file() else ()),
                        *tuple(v for v in ("wrong_padding", "wrong_skip") if (ROOT / "tasks" / name / f"model_new_{v}.py").is_file()),
                        *(("chain_replay",) if name == "eagle_gdn_tree_recurrence" and backend == "triton" else ()))
                for variant in selected_variants:
                    destination = root / name / item.id / backend / variant
                    result = evaluate_case(name, item.id, backend, variant, destination,
                                           device=device, purpose=purpose, cfg=config, cold_start=cold_start)
                    records.append(result)
                    with (root / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
                    print(json.dumps(result, allow_nan=False), flush=True)
    counts = {status: sum(row["suite/status"] == status for row in records)
              for status in ("passed", "failed", "unavailable")}
    # Controls with nothing to test (no skipped tokens, no rank padding) stay
    # "unavailable" rows, but they are planned and do not make the run partial.
    not_applicable = sum(row["suite/status"] == "unavailable"
                         and str(row.get("suite/reason") or "").startswith(NOT_APPLICABLE_PREFIX)
                         for row in records)
    report = {"schema": "progressive-task-suite-v1", "purpose": purpose, "device": device,
              "case_set": case_set, "counts": counts, "not_applicable": not_applicable, "rows": records,
              "status": "fail" if counts["failed"] else
                        ("partial" if counts["unavailable"] > not_applicable else "pass"),
              "limitations": ["Captured cases only; unavailable is never passed.",
                              "Name attribution is a heuristic, not proof of authorship.",
                              "Benchmark rates are diagnostics, not reward inputs."]}
    (root / "suite.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
