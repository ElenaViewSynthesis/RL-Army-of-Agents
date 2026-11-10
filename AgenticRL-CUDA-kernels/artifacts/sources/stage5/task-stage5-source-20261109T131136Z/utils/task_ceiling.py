"""Opt-in cuBLAS/GEMM diagnostic ceiling; never used by compute_reward."""

import argparse
import json
import statistics
import subprocess
from pathlib import Path


def unavailable(reason):
    return {"ceiling/available": False, "ceiling/unavailable_reason": reason}, {}


def measure(seed=0):
    from utils.backend import DEVICE

    if DEVICE.split(":")[0] != "cuda":
        return unavailable("CUDA unavailable")
    try:
        import torch
    except ImportError:
        return unavailable("Torch unavailable")
    from utils.benchmark import _time_calls
    from utils.verification import build_inputs, check_equal, initialize_models

    if not torch.cuda.is_available():
        return unavailable("CUDA unavailable")
    if not hasattr(torch.backends.cuda, "preferred_blas_library"):
        return unavailable("Cannot explicitly select the cuBLAS provider on this Torch version")
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(seed)
    reference, generated = initialize_models()
    reference_inputs, generated_inputs = build_inputs()
    if len(reference_inputs) != 3:
        return unavailable("Ceiling supports the A,B,bias Tensor Core GEMM contract only")
    a, b, bias = reference_inputs
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
            or a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype
            or bias.dtype != torch.float32 or bias.shape != (b.shape[1],)):
        return unavailable("Input contract does not match fp16/bf16 GEMM with fp32 output/bias")
    with torch.no_grad():
        try:
            vendor_output = torch.mm(a, b, out_dtype=torch.float32)
        except (TypeError, RuntimeError) as exc:
            return unavailable(f"cuBLAS fp32-output mm unavailable: {type(exc).__name__}: {exc}"[:512])
        expected = reference(*reference_inputs)
        check_equal(vendor_output + bias, expected)
        check_equal(generated(*generated_inputs), expected)
        calls = {"cublas": lambda: torch.mm(a, b, out_dtype=torch.float32),
                 "generated": lambda: generated(*generated_inputs)}
        for call in calls.values():
            for _ in range(10):
                call()
        torch.cuda.synchronize()
        raw = {name: {"event_us": [], "wall_us": []} for name in calls}
        order = []
        for round_index in range(6):
            names = ("cublas", "generated") if round_index % 2 == 0 else ("generated", "cublas")
            order.append(",".join(names))
            for name in names:
                event_us, wall_us = _time_calls(calls[name], 10)
                raw[name]["event_us"].append(event_us)
                raw[name]["wall_us"].append(wall_us)
    cublas_us = statistics.median(raw["cublas"]["event_us"])
    generated_us = statistics.median(raw["generated"]["event_us"])
    if not all(value > 0 for values in raw.values() for value in values["event_us"]):
        return unavailable("Nonpositive CUDA event sample")
    import math
    if not all(math.isfinite(value) for values in raw.values()
               for kind in ("event_us", "wall_us") for value in values[kind]):
        return unavailable("Nonfinite benchmark sample")
    flops = 2 * a.shape[0] * b.shape[1] * a.shape[1]
    index = a.device.index
    props = torch.cuda.get_device_properties(index)
    result = {"ceiling/available": True, "ceiling/unavailable_reason": None,
              "ceiling/provider": str(torch.backends.cuda.preferred_blas_library()),
              "ceiling/definition": "cuBLAS GEMM only, fp32 output; generated includes bias; 2MNK FLOPs",
              "ceiling/warmup_iters": 10, "ceiling/rounds": 6, "ceiling/iters_per_round": 10,
              "ceiling/seed": seed, "ceiling/dtype": str(a.dtype),
              "ceiling/m": a.shape[0], "ceiling/n": b.shape[1], "ceiling/k": a.shape[1],
              "ceiling/flops": flops, "ceiling/order": "|".join(order),
              "ceiling/cublas_event_us_median": cublas_us,
              "ceiling/generated_event_us_median": generated_us,
              "ceiling/cublas_tflops": flops / (cublas_us * 1e6),
              "ceiling/generated_tflops": flops / (generated_us * 1e6),
              "ceiling/cublas_fraction": cublas_us / generated_us,
              "ceiling/device_name": props.name,
              "ceiling/device_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
              "ceiling/device_capability": f"{props.major}.{props.minor}",
              "ceiling/torch_version": torch.__version__,
              "ceiling/cuda_version": torch.version.cuda}
    try:
        version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=30)
        result["ceiling/nvcc_version"] = version.stdout.strip() if version.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result["ceiling/nvcc_version"] = None
    return result, {"samples": raw, "order": order}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-json", type=Path, required=True)
    args = parser.parse_args()
    record, raw = measure(args.seed)
    args.samples_json.parent.mkdir(parents=True, exist_ok=True)
    args.samples_json.write_text(json.dumps({"summary": record, "raw": raw,
                                            "diagnostic_only": True}, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
