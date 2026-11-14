#!/usr/bin/env python3
"""Compare device time and attribute generated-model kernels to their source."""

import argparse
import json
import math
from pathlib import Path

from utils.backend import BACKEND_CHOICES, DEVICE, agent_kernel_names, detect_backend, get_spec
from utils.provenance import summarize_events


def get_prof_ctx():
    import torch

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )


def initialize_models():
    from utils.verification import build_inputs, initialize_models as _initialize_models

    torch_model, cuda_model = _initialize_models()
    torch_inputs, cuda_inputs = build_inputs()
    return torch_model, cuda_model, torch_inputs, cuda_inputs


def transform_tensors(tensors, fn):
    from utils.verification import transform_tensors as transform
    return transform(tensors, fn)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--single-run", type=str,
                        help="Run once: torch_baseline,torch_compile,cuda_extension")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    parser.add_argument("--json", action="store_true", help="Append a flat JSON device-time/provenance record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-dir", type=Path, help="Opt-in Chrome traces; diagnostic timings only")
    parser.add_argument("--order", choices=("gen,eager,compile", "eager,compile,gen"),
                        default="gen,eager,compile", help="Diagnostic session order")
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def benchmark_model(model, inputs, warmup_iters, run_iters, return_events=False, *, trace_path=None):
    import torch

    with torch.no_grad():
        for _ in range(warmup_iters):
            model(*inputs)
        torch.cuda.synchronize()
        with get_prof_ctx() as ctx:
            for _ in range(run_iters):
                model(*inputs)
            torch.cuda.synchronize()
    # CPU operator scopes also carry CUDA time and would double-count kernels.
    events = [(e.name, e.device_time) for e in ctx.events() if e.device_type.name == "CUDA"]
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.export_chrome_trace(str(trace_path))
    # No retry: retain the empty capture as evidence, never a zero latency.
    timing = sum(duration for _, duration in events) / run_iters if not capture_reason(events) else None
    return (timing, events) if return_events else timing


def capture_reason(events):
    if not events:
        return "empty_profile"
    if (any(not math.isfinite(duration) or duration < 0 for _, duration in events)
            or not math.isfinite(sum(duration for _, duration in events))):
        return "invalid_profile"
    return "nonpositive_profile" if sum(duration for _, duration in events) <= 0 else None


def prime_profiler():
    import torch

    counts = []
    for _ in range(2):
        torch.cuda.synchronize()
        with get_prof_ctx() as ctx:
            torch.cuda.synchronize()
        counts.append(sum(event.device_type.name == "CUDA" for event in ctx.events()))
    return counts


def profile_models(models, inputs, backend, iters, *, order="gen,eager,compile", trace_dir=None,
                   agent_names=(), prime=True, samples_dir=None):
    """One capture per session, retaining evidence even if another session is empty.

    samples_dir, when given, receives the verbatim per-event lists as a JSON
    sidecar; the record itself carries only per-kernel aggregates.
    """
    primer_events = prime_profiler() if prime else []
    captures = {}
    for session in order.split(","):
        trace = Path(trace_dir) / f"{session}.json" if trace_dir is not None else None
        captures[session] = benchmark_model(models[session], inputs[session],
                                           get_spec(backend).profile_warmup_iters,
                                           iters, True, trace_path=trace)
    failures = {session: capture_reason(capture[1]) for session, capture in captures.items()}
    summary = summarize_events(captures["gen"][1] if not failures["gen"] else [],
                               agent_names, backend, iters)
    reason = next((f"{failure}:{session}" for session, failure in failures.items() if failure), None)
    summary.update(timing_measured=reason is None,
                   timing_reason=reason,
                   t_eager_us=captures["eager"][0], t_compile_us=captures["compile"][0],
                   t_gen_us=captures["gen"][0])
    if failures["gen"]:
        summary["provenance"] = None
        summary.update({"provenance/"+s: None for s in ("agent","library","torch","memory")})
    summary.update({f"profile/{s}_events": len(captures[s][1]) for s in captures})
    # Per-kernel aggregates keep the stdout record small (run_task keeps only
    # its last 4 MiB); the verbatim per-event list goes to a sidecar file.
    summary.update({f"profile/{s}_kernel_samples": json.dumps(kernel_aggregates(captures[s][1]),
                                                             allow_nan=False) for s in captures})
    summary["profile/kernel_samples_path"] = write_kernel_samples(
        samples_dir, {s: captures[s][1] for s in captures}, order, iters)
    # JSON strings keep the profiler's record flat; names are verbatim.
    summary["profile/eager_kernel_names"] = json.dumps(sorted({n for n, _ in captures["eager"][1]}))
    summary["profile/order"] = order
    summary["profile/timing_method"] = ("cuda-leaf-sum-two-empty-prime-v1" if prime
                                        else "cuda-leaf-sum-unprimed-v1")
    summary["profile/primer_event_counts"] = json.dumps(primer_events)
    summary["profile/diagnostic"] = trace_dir is not None or order != "gen,eager,compile" or not prime
    if trace_dir is not None:
        summary.update({f"profile/{s}_trace": str(Path(trace_dir) / f"{s}.json") for s in captures})
    return summary


def _json_duration(value):
    return value if math.isfinite(value) else str(value)


def kernel_aggregates(events):
    """[[name, calls, total_us], ...] in first-seen order; non-finite totals as strings."""
    totals, calls = {}, {}
    for name, duration in events:
        totals[name] = totals.get(name, 0.0) + duration
        calls[name] = calls.get(name, 0) + 1
    return [[name, calls[name], _json_duration(totals[name])] for name in totals]


def write_kernel_samples(samples_dir, sessions, order, iters):
    """Write the verbatim per-event lists; return the path, or None if not written."""
    if samples_dir is None:
        return None
    try:
        from uuid import uuid4
        directory = Path(samples_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid4().hex}.json"
        path.write_text(json.dumps({
            "schema": "profile-kernel-samples-v1", "order": order, "iters": iters,
            "sessions": {s: [[name, _json_duration(d)] for name, d in events]
                         for s, events in sessions.items()}}, allow_nan=False) + "\n")
        return str(path)
    except OSError:
        return None


def print_results(torch_time, compile_time, cuda_time):
    # Deliberately byte-identical to upstream, for existing log consumers.
    msg = (
        f"Torch Baseline: {torch_time:.3f}us, "
        f"Torch Compile: {compile_time:.3f}us, "
        f"CUDA Extension: {cuda_time:.3f}us"
    )
    print(msg)


def run_single(targets, torch_model, cuda_model, torch_inputs, cuda_inputs):
    import torch

    allowed = {"torch_baseline", "torch_compile", "cuda_extension"}
    if set(targets) - allowed:
        raise ValueError(f"Unknown targets: {set(targets) - allowed}")
    with torch.no_grad():
        if "torch_baseline" in targets:
            torch_model(*torch_inputs)
        if "torch_compile" in targets:
            torch.compile(torch_model)(*torch_inputs)
        if "cuda_extension" in targets:
            cuda_model(*cuda_inputs)
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()
    print("[DONE] single-run completed")


def main():
    args = parse_args()
    backend = detect_backend(Path.cwd()) if args.backend == "auto" else args.backend
    # CPU interpreter correctness cannot supply device timing or provenance.
    # Do not import a large GPU runtime merely to report a CPU timing skip.
    if not args.single_run and DEVICE.split(":")[0] == "cpu":
        print("[SKIP] GPU timing/provenance not measured on CPU")
        if args.json:
            print(json.dumps({"timing_measured": False, "t_eager_us": None,
                              "t_compile_us": None, "t_gen_us": None, "provenance": None,
                              **{"provenance/" + k: None for k in ("agent", "library", "torch", "memory")}},
                             allow_nan=False))
        return
    import torch
    torch.manual_seed(args.seed)
    torch_model, cuda_model, torch_inputs, cuda_inputs = initialize_models()
    if args.single_run:
        targets = [x.strip() for x in args.single_run.split(",") if x.strip()]
        run_single(targets, torch_model, cuda_model, torch_inputs, cuda_inputs)
        return
    # JIT backends autotune on early calls; that must finish before timing.
    summary = profile_models(
        {"gen": cuda_model, "eager": torch_model, "compile": torch.compile(torch_model)},
        {"gen": cuda_inputs, "eager": torch_inputs, "compile": torch_inputs}, backend, args.iters,
        order=args.order, trace_dir=args.trace_dir, agent_names=agent_kernel_names(Path.cwd(), backend),
        samples_dir=Path.cwd() / ".artifacts" / "profile-samples")
    if summary["timing_measured"]:
        print_results(summary["t_eager_us"], summary["t_compile_us"], summary["t_gen_us"])
    else:
        print("[SKIP] " + summary["timing_reason"])
    if args.json:
        print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
