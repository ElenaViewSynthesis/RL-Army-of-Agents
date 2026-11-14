"""A plain CUDA workload for always-on tracers (Nsight Perf SDK, nsys): diagnostic only.

Run from a scaffolded task workdir with CUDA_AGENT_TASK, CUDA_AGENT_TASK_CASE and
CUDA_AGENT_DEVICE=cuda set, as the suite does. It verifies the generated model
once, then runs warm-up and timed forwards, each timed forward inside an NVTX
push/pop range named agenticrl_generated (the Nsight Perf SDK does not support
NVTX start/end ranges). It uses no torch.profiler, so it does not compete with
an external tracer for CUPTI. The CUDA-event time it prints is a diagnostic
wall-clock figure for the whole loop, never a reward input.

    python -m scripts.nsight_perf_workload --iters 200 --warmup 20
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--range", default="agenticrl_generated")
    args = parser.parse_args()
    if args.iters <= 0 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup non-negative")
    sys.path.insert(0, os.getcwd())  # the workdir's model.py and model_new.py
    import torch

    from utils.verification import build_inputs, initialize_models

    torch.manual_seed(0)
    reference, generated = initialize_models()
    torch_inputs, inputs = build_inputs()
    with torch.no_grad():
        expected, actual = reference(*torch_inputs), generated(*inputs)
        torch.cuda.synchronize()
        verified = bool(torch.allclose(actual.float(), expected.float(), rtol=1e-2, atol=1e-2))
        for _ in range(args.warmup):
            generated(*inputs)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            torch.cuda.nvtx.range_push(args.range)
            generated(*inputs)
            torch.cuda.nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
    print(json.dumps({"diagnostic_only": True, "task": os.environ.get("CUDA_AGENT_TASK"),
                      "case": os.environ.get("CUDA_AGENT_TASK_CASE"), "verified": verified,
                      "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                      "iters": args.iters, "warmup": args.warmup, "nvtx_range": args.range,
                      "loop_ms": start.elapsed_time(end), "per_forward_us": 1000 * start.elapsed_time(end) / args.iters}))
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
