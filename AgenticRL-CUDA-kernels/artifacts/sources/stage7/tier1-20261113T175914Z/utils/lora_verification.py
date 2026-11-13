"""Approved Stage 6 relative verification. Torch is imported only on use.

The subprocess exit status gates verification. Reports only explain that decision;
candidate-writable report files are never read back to compute rewards.
"""


def bit_equal(actual, expected):
    import torch
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8)))


def evaluate_outputs(name, actual, expected, inputs):
    import torch
    from utils.task_catalog import get_task
    assert get_task(name).stage == 6, "Relative bounds are Stage 6 only"
    metadata = name == "lora_segment_metadata"
    shrink = "shrink" in name
    dtype = inputs[0].dtype if shrink or "fused" in name else inputs[2].dtype if not metadata else None
    if not metadata:
        assert dtype in (torch.float16, torch.bfloat16), "fp16/bf16 activations required"
    c = .01 if dtype == torch.bfloat16 else .005
    records = []
    idx = None if metadata else inputs[2] if shrink else inputs[4] if "fused" in name else inputs[3]
    y = None if metadata or shrink else inputs[3] if "fused" in name else inputs[2]

    def walk(a, b, path):
        assert type(a) is type(b), f"{path}: output type mismatch"
        if isinstance(b, torch.Tensor):
            assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, f"{path}: tensor metadata mismatch"
            if not b.is_floating_point():
                passed = torch.equal(a, b)
                records.append({"output": path, "kind": "integer_exact", "passed": passed})
                return
            active = torch.ones_like(b, dtype=torch.bool)
            skipped_exact = True
            if y is not None:
                assert b.shape == y.shape, "y output shape mismatch"
                skipped = idx == -1
                skipped_exact = bit_equal(a[skipped], y[skipped])
                active = (idx != -1)[:, None].expand_as(b)
            aa, bb = a[active].double(), b[active].double()
            finite = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
            ref_max = (bb.abs().max().item() if bb.numel() else 0.) if bool(torch.isfinite(bb).all()) else None
            error = (aa - bb).abs()
            max_error = error.max().item() if error.numel() and finite else 0. if finite else None
            bound = c * ref_max if ref_max is not None else None
            zero_exact = torch.equal(aa, bb) if ref_max == 0 else True
            passed = bool(skipped_exact and finite and zero_exact and max_error <= bound)
            normalized = error / ref_max if ref_max else error
            quantiles = (torch.quantile(normalized, torch.tensor([.5, .99], dtype=torch.float64, device=b.device)).tolist()
                         if normalized.numel() and finite else [None, None])
            records.append({"output": path, "kind": "relative", "coefficient": c,
                            "checked_elements": bb.numel(), "max_abs_ref": ref_max,
                            "max_abs_error": max_error, "bound": bound,
                            "ratio_to_bound": max_error / bound if finite and bound else None,
                            "p50_normalized_error": quantiles[0], "p99_normalized_error": quantiles[1],
                            "finite": finite, "skipped_rows_bit_exact": skipped_exact,
                            "zero_reference_exact": bool(zero_exact), "passed": passed})
        elif isinstance(b, (tuple, list)):
            assert len(a) == len(b), f"{path}: output length mismatch"
            for i, (av, bv) in enumerate(zip(a, b)):
                walk(av, bv, f"{path}/{i}")
        elif isinstance(b, dict):
            assert a.keys() == b.keys(), f"{path}: dictionary mismatch"
            for key in b:
                walk(a[key], b[key], f"{path}/{key}")
        else:
            assert a == b, f"{path}: scalar mismatch"
    walk(actual, expected, "output")
    return {"task": name, "activation_dtype": str(dtype), "outputs": records,
            "passed": all(row["passed"] for row in records)}


def check_outputs(name, actual, expected, inputs):
    report = evaluate_outputs(name, actual, expected, inputs)
    assert report["passed"], str(report)
    return report


def main():
    import argparse
    import json
    import os
    from pathlib import Path
    import torch
    from utils import verification as v
    from utils.task_catalog import get_task
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    name = os.environ['CUDA_AGENT_TASK']
    assert get_task(name).stage == 6, "Relative bounds are Stage 6 only"
    target = Path('.cache/relative-error-bounds.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "stage6-relative-bounds-v1", "task": name, "checks": [], "passed": False}
    def save():
        target.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    save()
    torch.manual_seed(args.seed)
    try:
        reference, candidate = v.initialize_models()
        with torch.no_grad():
            for i in range(5):
                trusted, supplied = v.build_inputs()
                expected = reference(*trusted)
                with v.block_torch_functional():
                    actual = candidate(*supplied)
                report = evaluate_outputs(name, actual, expected, trusted)
                payload['checks'].append(report)
                save()
                assert report['passed'], str(report)
                print(f'[PASS] check {i + 1}/5')
        payload['passed'] = True
        save()
        print('[PASS] verify success')
    except Exception as exc:
        payload['error'] = f'{type(exc).__name__}: {exc}'[:4096]
        save()
        raise


if __name__ == '__main__':
    main()
