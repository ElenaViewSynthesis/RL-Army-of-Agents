#!/usr/bin/env python3
"""Compare device time and attribute generated-model kernels to their source."""

import argparse
import json
from pathlib import Path

from utils.backend import DEVICE, agent_kernel_names, detect_backend
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
    parser.add_argument("--backend", choices=("auto", "cuda", "triton"), default="auto")
    parser.add_argument("--json", action="store_true", help="Append a flat JSON device-time/provenance record")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def benchmark_model(model, inputs, warmup_iters, run_iters, return_events=False):
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
    timing = sum(duration for _, duration in events) / run_iters
    return (timing, events) if return_events else timing


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
    warmup_iters = 20 if backend == "triton" else 5
    cuda_time, events = benchmark_model(cuda_model, cuda_inputs, warmup_iters, args.iters, True)
    torch_time = benchmark_model(torch_model, torch_inputs, warmup_iters, args.iters)
    compile_time = benchmark_model(torch.compile(torch_model), torch_inputs, warmup_iters, args.iters)
    print_results(torch_time, compile_time, cuda_time)
    if args.json:
        summary = summarize_events(events, agent_kernel_names(Path.cwd(), backend), backend, args.iters)
        summary.update(timing_measured=True, t_eager_us=torch_time, t_compile_us=compile_time)
        print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
