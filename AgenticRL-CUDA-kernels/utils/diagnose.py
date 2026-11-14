import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from utils.backend import BACKENDS, agent_kernel_names, detect_backend, get_spec
from utils.rollout_env import bootstrap_script

TOOLS = {"sanitizer": "compute-sanitizer", "nsys": "nsys", "ncu": "ncu"}
CHECKS = ("memcheck", "racecheck", "synccheck", "initcheck")
RANGE = "agenticrl_generated"
# NVTX ranges pushed by utils.diagnostic_worker (Part 2 of NSIGHT_profiling_task.md).
# RANGE keeps its name and content: exactly one generated forward.
ROLLOUT_RANGE = "agenticrl_rollout"
WARMUP_RANGE = "agenticrl_warmup"
EAGER_RANGE = "agenticrl_eager"
COMPILE_RANGE = "agenticrl_compile"
NVTX_RANGES = (ROLLOUT_RANGE, WARMUP_RANGE, EAGER_RANGE, COMPILE_RANGE, RANGE)
ENVIRONMENT = (
    "PATH", "HOME", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH",
    "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "TORCH_CUDA_ARCH_LIST",
    "MAX_JOBS", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    # A task workdir's model.py reads its trusted case from these; absent for AXPBY.
    "CUDA_AGENT_TASK", "CUDA_AGENT_TASK_CASE",
)

# Named Nsight Systems profiles; tool_command composes them in a fixed order.
# Costs are expectations to confirm on the H200, not measurements.
NSYS_PROFILES = {
    # Today's validated capture (20261108T200118Z) of exactly one generated
    # forward. Cost: the baseline.
    "trace": "One generated forward under CUDA and NVTX tracing.",
    # The whole diagnostic run (warm-up, eager, torch.compile, generated) plus
    # cuBLAS, cuDNN and OS-runtime tracing. Cost: one torch.compile, a longer
    # capture window and per-call library records; larger reports.
    "stages": "Whole diagnostic run with cuBLAS, cuDNN and OS-runtime tracing.",
    # nsys's own NVTX injection around PyTorch functions and autograd ops.
    # Cost: host overhead per op; needs a PyTorch build nsys can hook.
    "pytorch": "PyTorch function and autograd-shape NVTX ranges injected by nsys.",
    # Host stacks for every CUDA call and the Python line that launched it.
    # Cost: high (stack unwinding per call); never on a timed row.
    "backtrace": "CUDA and Python backtraces for every kernel launch; high overhead.",
    # Hardware metric sampling of the assigned device only. Cost: a sampling
    # stream; may need counter permission on some hosts.
    "gpu-metrics": "SM, Tensor Core and DRAM activity over time for the assigned GPU.",
    # Per-node kernels inside cudaGraphLaunch. Cost: small.
    "graphs": "Per-node kernel records inside CUDA graph launches.",
}
# Post-capture nsys reports (NSIGHT_PROFILING.md sections 2-3). They read the
# finished capture.sqlite, never touch the GPU, and never change a capture's
# status: each runs as its own bounded stage recorded under report["nsys_reports"].
NSYS_STATS_REPORTS = ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum", "cuda_gpu_mem_time_sum")
NSYS_ANALYZE_RULES = ("gpu_gaps", "gpu_time_util", "cuda_api_sync", "cuda_memcpy_sync")
NSYS_REPORT_LIMIT_S = 120
NCU_PRESETS = {
    "basic": ("--set", "basic"),
    "roofline": ("--section", "SpeedOfLight", "--section", "MemoryWorkloadAnalysis",
                 "--section", "Occupancy"),
    "launch": ("--section", "LaunchStats"),
}


def parse_profiles(value):
    names = tuple(dict.fromkeys(part.strip() for part in str(value).split(",") if part.strip()))
    if not names or set(names) - set(NSYS_PROFILES):
        raise ValueError(f"Unknown or empty nsys profile list: {value!r}")
    return names


def nsys_capture_range(profiles):
    return ROLLOUT_RANGE if "stages" in profiles else RANGE


def diagnostic_environment(workdir, tool):
    env = {key: os.environ[key] for key in ENVIRONMENT if key in os.environ}
    env.update(CUDA_AGENT_DEVICE=os.environ.get("CUDA_AGENT_DEVICE", "cuda"),
               TRITON_INTERPRET="0", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled",
               WEAVE_DISABLED="true", XDG_CACHE_HOME=str(workdir / ".cache"))
    for spec in BACKENDS.values():
        for key, subdir in spec.cache_env:
            env[key] = str(workdir / ".cache" / subdir)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(workdir / ".cache" / "inductor")
    if tool == "sanitizer":
        env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    if tool == "nsys":
        env["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
    return env


def run_process(argv, workdir, env, deadline, out, label):
    stdout = out / f"{label}.stdout.log"
    stderr = out / f"{label}.stderr.log"
    record = {"argv": argv, "command": shlex.join(argv), "stdout": str(stdout),
              "stderr": str(stderr), "returncode": None, "timed_out": False,
              "wall_s": None}
    started = time.monotonic()
    if started >= deadline:
        record.update(timed_out=True, wall_s=0.0)
        return record
    with stdout.open("wb") as output, stderr.open("wb") as error:
        try:
            proc = subprocess.Popen(argv, cwd=workdir, env=env, stdout=output,
                                    stderr=error, start_new_session=True)
        except OSError as exc:
            record["error_type"] = type(exc).__name__
            error.write(str(exc).encode("utf-8", "replace"))
        else:
            try:
                proc.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
            record["returncode"] = proc.returncode
    record["wall_s"] = time.monotonic() - started
    return record


def load_record(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def source_hashes(workdir):
    paths = [workdir / name for name in ("model.py", "model_new.py", "binding.cpp", "binding_registry.h")]
    paths += [path for path in (workdir / "kernels").rglob("*")
              if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".h")]
    return {str(path.relative_to(workdir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def kernel_selector(workdir, backend, kernel):
    if kernel is None:
        names = agent_kernel_names(workdir, backend)
        kernel = "|".join("(?:" + get_spec(backend).symbol_pattern.format(name=re.escape(name)) + ")"
                          for name in names)
    if not kernel:
        raise ValueError("A nonempty kernel selector is required for Nsight Compute")
    re.compile(kernel)
    return kernel


def gpu_metrics_device(env, run=subprocess.run):
    """nsys GPU index of the assigned device, resolved as utils.gpu_lock does.

    The logical ordinal in CUDA_AGENT_DEVICE is mapped through
    CUDA_VISIBLE_DEVICES; a UUID token is resolved to its nvidia-smi index.
    Returns None when the device cannot be resolved; never a guessed 0.
    """
    from utils.gpu_lock import selected_device_token
    token = selected_device_token(env)
    if token is None:
        return None
    if token.isdigit():
        return token
    binary = shutil.which("nvidia-smi", path=env.get("PATH"))
    if binary is None:
        return None
    try:
        query = run([binary, "--query-gpu=index", "--format=csv,noheader", "-i", token],
                    env=env, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    return rows[0] if len(rows) == 1 and rows[0].isdigit() else None


def tool_command(tool, binary, worker, out, check, kernel, *, profiles=("trace",), preset="basic",
                 metrics_device=None):
    if tool == "sanitizer":
        return [binary, "--tool", check, "--error-exitcode", "86", "--target-processes",
                "application-only", "--log-file", str(out / "sanitizer.log"), *worker]
    if tool == "nsys":
        profiles = parse_profiles(",".join(profiles))
        trace = "cuda,nvtx,cublas,cudnn,osrt" if "stages" in profiles else "cuda,nvtx"
        extra = []
        if "pytorch" in profiles:
            extra.append("--pytorch=functions-trace,autograd-shapes-nvtx")
        if "backtrace" in profiles:
            extra += ["--cudabacktrace=all", "--python-backtrace=cuda"]
        if "gpu-metrics" in profiles:
            if metrics_device is None:
                raise ValueError("gpu-metrics needs the assigned device index")
            extra.append(f"--gpu-metrics-devices={metrics_device}")
        if "graphs" in profiles:
            extra.append("--cuda-graph-trace=node")
        # profiles=("trace",) reproduces the 2026-11-08 command byte for byte.
        # --cudabacktrace and --python-backtrace need CPU sampling enabled (Nsight
        # Systems CLI guide); every other profile keeps sampling off.
        sample = "process-tree" if "backtrace" in profiles else "none"
        return [binary, "profile", f"--trace={trace}", f"--sample={sample}", "--cpuctxsw=none",
                "--discard-environment=true", "--capture-range=nvtx",
                f"--nvtx-capture={nsys_capture_range(profiles)}", "--capture-range-end=stop",
                *extra, "--export=sqlite", "--output", str(out / "capture"), *worker]
    if preset not in NCU_PRESETS:
        raise ValueError(f"Unknown Nsight Compute preset {preset!r}")
    return [binary, "--clock-control", "none", "--nvtx", "--nvtx-include", RANGE + "/", "--kernel-name-base", "demangled",
            "--kernel-name", "regex:" + kernel, "--launch-count", "1", *NCU_PRESETS[preset],
            "--export", str(out / "capture"), *worker]


def nsys_report_commands(binary, out):
    """(label, argv) for each post-capture report; CSV goes to the stage's stdout log."""
    source = str(out / "capture.sqlite")
    commands = [(f"stats_{name}", [binary, "stats", "--report", name, "--format", "csv", source])
                for name in NSYS_STATS_REPORTS]
    commands += [(f"analyze_{rule}", [binary, "analyze", "--rule", rule, "--format", "csv", source])
                 for rule in NSYS_ANALYZE_RULES]
    return commands


def gpu_metrics_devices_help(binary):
    """The device IDs nsys itself reports for --gpu-metrics-devices (the open
    question in NSIGHT_PROFILING.md section 6): recorded, never interpreted."""
    return [binary, "profile", "--gpu-metrics-devices=help"]


def run_diagnostic(tool, workdir, out, *, backend="auto", check="memcheck", kernel=None,
                   timeout_s=600, seed=0, negative_control=False, profile="trace", preset="basic",
                   nsys_reports=True):
    if tool not in TOOLS or check not in CHECKS or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Known tool/check and positive timeout required")
    profiles = parse_profiles(profile)
    if profiles != ("trace",) and tool != "nsys":
        raise ValueError("--profile applies only to Nsight Systems")
    if preset not in NCU_PRESETS or (preset != "basic" and tool != "ncu"):
        raise ValueError("--preset applies only to Nsight Compute and must be a known preset")
    if negative_control and (tool != "sanitizer" or check != "memcheck" or backend != "triton"):
        raise ValueError("Invalid-access control is restricted to Triton sanitizer memcheck")
    if kernel is not None and tool != "ncu":
        raise ValueError("--kernel is only supported by Nsight Compute")
    workdir, out = Path(workdir).resolve(), Path(out).resolve()
    if not workdir.is_dir():
        raise ValueError("The task workdir must exist")
    backend = detect_backend(workdir) if backend == "auto" else backend
    get_spec(backend)
    if tool == "ncu":
        kernel = kernel_selector(workdir, backend, kernel)
    out.mkdir(parents=True, exist_ok=False)
    (workdir / ".cache").mkdir(exist_ok=True)
    started = time.monotonic()
    deadline = started + timeout_s
    env = diagnostic_environment(workdir, tool)
    report = {"schema": "gpu-diagnostics-v1", "diagnostic_only": True, "stages": [],
              "sources": source_hashes(workdir), "worker": None, "gpu_probe": None}
    if tool in ("nsys", "ncu"):
        # Lets the reviewer split kernel time into authored/library/torch by name.
        report["agent_kernel_names"] = sorted(agent_kernel_names(workdir, backend))
    result = {"diag/tool": tool, "diag/check": check if tool == "sanitizer" else None,
              "diag/backend": backend, "diag/status": "failed", "diag/reason": None,
              "diag/diagnostic_only": True, "diag/evidence_reviewed": False,
              "diag/negative_control": negative_control, "diag/seed": seed,
              "diag/clock_control": "none" if tool == "ncu" else None,
              "diag/kernel_selector": None, "diag/gpu_uuid": None, "diag/gpu_name": None,
              "diag/nsys_profiles": ",".join(profiles) if tool == "nsys" else None,
              "diag/ncu_preset": preset if tool == "ncu" else None,
              "diag/reported_errors": None, "diag/kernel_count": None,
              "diag/wall_s": None, "diag/report_json": str(out / "report.json")}
    for name in ("diagnose.py", "diagnostic_worker.py", "diagnostic_fault.py"):
        path = Path(__file__).with_name(name)
        report.setdefault("harness_sha256", {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def finish(status, reason):
        result.update({"diag/status": status, "diag/reason": reason,
                       "diag/wall_s": time.monotonic() - started})
        report["result"] = result
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        from utils.diagnostic_review import review_report
        return review_report(out / "report.json")["result"]

    def execute(argv, label, limit=None, into="stages"):
        stage = run_process(argv, workdir, env, min(deadline, time.monotonic() + limit)
                            if limit is not None else deadline, out, label)
        stage["label"] = label
        report.setdefault(into, []).append(stage)
        return stage

    binary = shutil.which(TOOLS[tool])
    if binary is None:
        return finish("unavailable", "tool_not_installed")
    if env["CUDA_AGENT_DEVICE"].split(":")[0] != "cuda":
        return finish("unavailable", "cuda_required")
    version = execute([binary, "--version"], "version", 30)
    if version["timed_out"]:
        return finish("timeout", "version_timeout")
    if version["returncode"] != 0:
        return finish("unavailable", "tool_version_failed")
    report["tool_version"] = Path(version["stdout"]).read_text(errors="replace").strip()
    probe = execute([sys.executable, "-c", bootstrap_script(
        "utils.diagnostic_worker", ["--mode", "probe", "--backend", backend,
                                    "--evidence", str(out / "gpu.json")], workdir)], "gpu", 60)
    report["gpu_probe"] = gpu = load_record(out / "gpu.json")
    if probe["timed_out"]:
        return finish("timeout", "gpu_probe_timeout")
    if probe["returncode"] != 0 or not isinstance(gpu, dict):
        return finish("failed", "gpu_probe_failed")
    if not gpu.get("available"):
        return finish("unavailable", gpu.get("reason") or "cuda_unavailable")
    if not gpu.get("completed"):
        return finish("failed", "gpu_probe_incomplete")
    result.update({"diag/gpu_uuid": gpu["gpu_uuid"], "diag/gpu_name": gpu["gpu_name"]})
    if backend == "cuda" and not (workdir / "cuda_extension.so").is_file():
        compile_stage = execute([sys.executable, "-c", bootstrap_script(
            "utils.compile", ["--backend", backend, "--seed", str(seed)], workdir)], "compile")
        if compile_stage["timed_out"]:
            return finish("timeout", "compile_timeout")
        if compile_stage["returncode"] != 0:
            return finish("failed", "compile_failed")
    if tool == "ncu":
        result["diag/kernel_selector"] = kernel
    mode = "invalid-access" if negative_control else "verify" if tool == "sanitizer" else "capture"
    worker_args = ["--mode", mode, "--backend", backend, "--seed", str(seed),
                   "--evidence", str(out / "worker.json")]
    if tool == "nsys" and "stages" in profiles:
        worker_args.append("--with-compile")
    worker = [sys.executable, "-c", bootstrap_script("utils.diagnostic_worker", worker_args, workdir)]
    metrics_device = None
    if tool == "nsys" and "gpu-metrics" in profiles:
        metrics_device = gpu_metrics_device(env)
        if metrics_device is None:
            return finish("unavailable", "gpu_metrics_device_unresolved")
        result["diag/nsys_gpu_metrics_device"] = metrics_device
        execute(gpu_metrics_devices_help(binary), "gpu_metrics_devices_help", 60, into="nsys_reports")
    stage = execute(tool_command(tool, binary, worker, out, check, result["diag/kernel_selector"],
                                 profiles=profiles, preset=preset, metrics_device=metrics_device), "tool")
    report["worker"] = worker_record = load_record(out / "worker.json")
    if stage["timed_out"]:
        return finish("timeout", "tool_timeout")
    if stage["returncode"] != 0:
        return finish("failed", "tool_failed")
    if not isinstance(worker_record, dict) or not worker_record.get("completed"):
        return finish("failed", "worker_incomplete")
    if tool == "nsys":
        if not all((out / name).is_file() and (out / name).stat().st_size > 0
                   for name in ("capture.nsys-rep", "capture.sqlite")):
            return finish("failed", "missing_trace")
        if nsys_reports:
            for label, argv in nsys_report_commands(binary, out):
                execute(argv, label, NSYS_REPORT_LIMIT_S, into="nsys_reports")
    if tool == "ncu":
        if not (out / "capture.ncu-rep").is_file() or (out / "capture.ncu-rep").stat().st_size == 0:
            return finish("failed", "missing_counter_report")
        exported = execute([binary, "--import", str(out / "capture.ncu-rep"),
                            "--csv", "--page", "raw"], "metrics")
        if exported["timed_out"]:
            return finish("timeout", "metrics_export_timeout")
        if exported["returncode"] != 0:
            return finish("failed", "metrics_export_failed")
    return finish("captured", "raw_evidence_requires_review")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=(*TOOLS, "review"))
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--backend", choices=("auto", *BACKENDS), default="auto")
    parser.add_argument("--check", choices=CHECKS, default="memcheck")
    parser.add_argument("--kernel")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--profile", default="trace",
                        help="nsys only: comma-separated profiles: " + ",".join(NSYS_PROFILES))
    parser.add_argument("--preset", choices=tuple(NCU_PRESETS), default="basic",
                        help="ncu only: basic (validated), roofline or launch")
    parser.add_argument("--no-nsys-reports", action="store_true",
                        help="nsys only: skip the post-capture nsys stats/analyze reports")
    args = parser.parse_args()
    if args.tool == "review":
        if args.report is None:
            parser.error("review requires --report")
        from utils.diagnostic_review import review_report
        result = review_report(args.report)["result"]
        print(json.dumps(result, allow_nan=False))
        return 1 if result["diag/status"] in ("failed", "timeout") else 0
    if args.out is None:
        parser.error("--out is required")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if args.kernel is not None and args.tool != "ncu":
        parser.error("--kernel requires ncu")
    try:
        profiles = parse_profiles(args.profile)
    except ValueError as exc:
        parser.error(str(exc))
    if profiles != ("trace",) and args.tool != "nsys":
        parser.error("--profile requires nsys")
    if args.preset != "basic" and args.tool != "ncu":
        parser.error("--preset requires ncu")
    if args.negative_control and (args.tool != "sanitizer" or args.check != "memcheck"
                                 or args.backend != "triton"):
        parser.error("--negative-control requires sanitizer memcheck and --backend triton")
    result = run_diagnostic(args.tool, args.workdir, args.out, backend=args.backend, check=args.check,
                            kernel=args.kernel, timeout_s=args.timeout_s, seed=args.seed,
                            negative_control=args.negative_control, profile=args.profile,
                            preset=args.preset, nsys_reports=not args.no_nsys_reports)
    print(json.dumps(result, allow_nan=False))
    return 1 if result["diag/status"] in ("failed", "timeout") else 0


if __name__ == "__main__":
    sys.exit(main())
