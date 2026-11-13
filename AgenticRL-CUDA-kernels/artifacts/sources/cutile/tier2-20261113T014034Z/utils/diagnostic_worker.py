import argparse
import json
from pathlib import Path

from utils.backend import DEVICE, get_spec


def write_record(path, record):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "verify", "capture", "invalid-access"), required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = get_spec(args.backend)
    record = {"diagnostic_only": True, "available": False, "completed": False,
              "mode": args.mode, "backend": args.backend, "reason": None}
    try:
        import torch
    except ImportError:
        record["reason"] = "torch_unavailable"
        write_record(args.evidence, record)
        return
    if DEVICE.split(":")[0] != "cuda" or not torch.cuda.is_available():
        record["reason"] = "cuda_unavailable"
        write_record(args.evidence, record)
        return
    device = torch.device(DEVICE)
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    record.update(available=True, gpu_name=props.name, gpu_index=torch.cuda.current_device(),
                  gpu_uuid=str(props.uuid) if getattr(props, "uuid", None) is not None else None,
                  compute_capability=f"{props.major}.{props.minor}",
                  negative_control=args.mode == "invalid-access")
    write_record(args.evidence, record)
    if args.mode == "probe":
        record["completed"] = True
        write_record(args.evidence, record)
        return
    torch.manual_seed(args.seed)
    if args.mode == "invalid-access":
        if args.backend != "triton":
            raise ValueError("The diagnostic invalid-access control requires Triton")
        from utils.diagnostic_fault import run
        run()
    else:
        from utils.verification import (
            block_torch_functional, build_inputs, check_equal, initialize_models,
        )
        eager, generated = initialize_models()
        with torch.no_grad():
            if args.mode == "verify":
                for _ in range(5):
                    eager_inputs, generated_inputs = build_inputs()
                    expected = eager(*eager_inputs)
                    with block_torch_functional():
                        actual = generated(*generated_inputs)
                    check_equal(actual, expected)
                torch.cuda.synchronize()
                record["verification_checks"] = 5
            else:
                eager_inputs, generated_inputs = build_inputs()
                expected = eager(*eager_inputs)
                for _ in range(spec.profile_warmup_iters):
                    with block_torch_functional():
                        actual = generated(*generated_inputs)
                torch.cuda.synchronize()
                check_equal(actual, expected)
                torch.cuda.nvtx.range_push("agenticrl_generated")
                try:
                    with block_torch_functional():
                        actual = generated(*generated_inputs)
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.nvtx.range_pop()
                check_equal(actual, expected)
                record.update(warmup_iters=spec.profile_warmup_iters,
                              capture_range="agenticrl_generated", captured_forwards=1,
                              verification_checks=2)
    record["completed"] = True
    write_record(args.evidence, record)
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
