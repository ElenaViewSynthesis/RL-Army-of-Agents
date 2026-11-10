"""Opt-in diagnostic profiler pass that writes actionable artifacts.

Run as a separate run_task stage after the reward profiler (and benchmark).
It profiles only the generated model, with CPU + CUDA activities, recorded
shapes and memory events, and writes into --out-dir:

  report.json   schema, inputs, GPU/software, source SHA-256, capture,
                provenance, benchmark samples, tool capability report
  kernels.json  per-kernel calls / total / mean / per-iteration µs and class
  trace.json    Chrome trace (GPU only; never written on CPU)

Profiler instrumentation perturbs execution, so none of these timings is a
reward input, and the reward never reads these files. The last stdout line
is a flat JSON record of artifact/* paths.
"""

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

from utils import artifact_report
from utils.provenance import PROVENANCE_RULESET
from utils.backend import BACKEND_CHOICES, DEVICE, agent_kernel_names, detect_backend, get_spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    parser.add_argument("--iters", type=int, default=10, help="Profiled iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bench-samples", type=Path,
                        help="benchmark --samples-json file to embed, when the benchmark ran")
    args = parser.parse_args(argv)
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def describe(value):
    """JSON-safe shape/dtype description of get_inputs() values."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (bool, int, str)) or value is None:
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__}


def record_inputs():
    if importlib.util.find_spec("torch") is None:
        return None, "torch is not installed"
    try:
        from model import get_inputs
        inputs = get_inputs()
        return describe(list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]), None
    except Exception as exc:  # A failing get_inputs is data here, not a crash.
        return None, f"get_inputs() failed: {type(exc).__name__}: {exc}"[:512]


def load_benchmark(path):
    if path is None:
        return {"enabled": False}
    if not path.is_file():
        return {"enabled": True, "available": False, "reason": "benchmark samples file missing"}
    try:
        return {"enabled": True, **json.loads(path.read_text())}
    except ValueError:
        return {"enabled": True, "available": False, "reason": "malformed benchmark samples file"}


def driver_version():
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"nvidia-smi unavailable: {type(exc).__name__}"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        return None, f"nvidia-smi exited {proc.returncode}"
    return lines[0], None


def gpu_metadata(torch):
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {"index": index, "name": torch.cuda.get_device_name(index),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": int(props.total_memory),
            "uuid": str(props.uuid) if hasattr(props, "uuid") else None}


def software_versions(torch=None):
    """Installed versions; the CUDA runtime is only read from an imported Torch."""
    from importlib import metadata

    def version(dist):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None
    driver, driver_reason = driver_version()
    return {"torch": torch.__version__ if torch is not None else version("torch"),
            "cuda_runtime": torch.version.cuda if torch is not None else None,
            "triton": version("triton"), "driver": driver, "driver_reason": driver_reason}


def capture_generated(torch, backend, iters, trace_path):
    """One CPU+CUDA profiler pass over the generated model; returns (events, capture)."""
    from utils.profiling import capture_reason, prime_profiler
    from utils.verification import build_inputs, initialize_models

    _, model = initialize_models()
    _, inputs = build_inputs()
    primer = prime_profiler()
    warmup = get_spec(backend).profile_warmup_iters
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True, with_stack=False) as ctx:
            for _ in range(iters):
                model(*inputs)
            torch.cuda.synchronize()
    all_events = list(ctx.events())
    # Leaf CUDA events only; CPU operator scopes would double-count kernels.
    events = [(e.name, e.device_time) for e in all_events if e.device_type.name == "CUDA"]
    ctx.export_chrome_trace(str(trace_path))
    return events, {"session": "generated", "iters": iters, "warmup_iters": warmup,
                    "activities": ["CPU", "CUDA"], "record_shapes": True,
                    "profile_memory": True, "primer_event_counts": primer,
                    "cuda_events": len(events), "cpu_events": len(all_events) - len(events),
                    "reason": capture_reason(events)}


def run(args):
    started = utc_now()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path.cwd()
    backend = detect_backend(workdir) if args.backend == "auto" else args.backend
    gpu = software = capture = trace = None
    rows = []
    reason = None
    if DEVICE.split(":")[0] == "cpu":
        reason = "CUDA_AGENT_DEVICE=cpu"
    else:
        import torch
        if not torch.cuda.is_available():
            reason = "torch.cuda.is_available() is False"
        else:
            torch.manual_seed(args.seed)
            gpu, software = gpu_metadata(torch), software_versions(torch)
            trace_path = out / "trace.json"
            try:
                events, capture = capture_generated(torch, backend, args.iters, trace_path)
                rows = artifact_report.kernel_table(events, agent_kernel_names(workdir, backend),
                                                    backend, args.iters)
                reason = capture["reason"]
            except Exception as exc:  # Diagnostics fail as data, never as a reward change.
                reason = f"capture failed: {type(exc).__name__}: {exc}"[:512]
                capture = {"session": "generated", "reason": reason}
            trace = str(trace_path) if trace_path.is_file() else None
    if software is None:
        software = software_versions()
    inputs, inputs_reason = record_inputs()
    cuda_events = capture.get("cuda_events") if capture else None
    report = artifact_report.build_report(
        run_id=args.run_id, task_id=workdir.name, backend=backend, started_at=started,
        finished_at=utc_now(), workdir=workdir, inputs=inputs, inputs_reason=inputs_reason,
        gpu=gpu, gpu_reason=reason if gpu is None else None, software=software,
        capture=capture, rows=rows, benchmark=load_benchmark(args.bench_samples),
        capabilities=artifact_report.tool_capabilities(cupti_events=cuda_events, cupti_reason=reason),
        trace=trace, trace_reason=reason if trace is None else None)
    kernels = {"schema_version": artifact_report.SCHEMA_VERSION, "run_id": args.run_id,
               "backend": backend, "measured": bool(rows) and reason is None,
               "reason": reason, "iters": args.iters,
               "ruleset": PROVENANCE_RULESET,
               "notes": list(artifact_report.NOTES), "kernels": rows}
    artifact_report.write_json(out / "kernels.json", kernels)
    artifact_report.write_json(out / "report.json", report)
    return {"artifact/report": str(out / "report.json"), "artifact/kernels": str(out / "kernels.json"),
            "artifact/trace": trace, "artifact/unavailable_reason": reason}


def main():
    record = run(parse_args())
    if record["artifact/unavailable_reason"]:
        print(f"[SKIP] diagnostic profile unavailable: {record['artifact/unavailable_reason']}")
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
