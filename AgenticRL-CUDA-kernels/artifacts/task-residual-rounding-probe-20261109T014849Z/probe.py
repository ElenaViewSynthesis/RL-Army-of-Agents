import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from triton.runtime import interpreter


root = Path("/mnt/c/Users/proxi/Documents/codex-9/CUDA-RL/AgenticRL-CUDA-kernels")
sys.path.insert(0, str(root))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bf16_bits(data):
    return torch.from_numpy(data.copy()).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def comparison(actual, expected):
    close = torch.isclose(actual, expected, rtol=1e-2, atol=1e-2)
    indices = (~close).nonzero()[:5]
    return {"exact_mismatches": int((actual != expected).sum()),
            "outside_approved_tolerance": int((~close).sum()),
            "maximum_absolute_difference": float((actual.float() - expected.float()).abs().max()),
            "examples": [{"index": row.tolist(), "actual": float(actual[tuple(row)]),
                          "expected": float(expected[tuple(row)])} for row in indices]}


reference_module = load("_rr_ref", root / "tasks/residual_rmsnorm/model.py")
kernel_module = load("_rr_kernel", root / "tasks/residual_rmsnorm/kernels/triton/residual_rmsnorm.py")
torch.manual_seed(0)
reference = reference_module.Model(*reference_module.get_init_inputs())
inputs = reference_module.get_inputs()
with torch.no_grad():
    expected = reference(*inputs)
original = interpreter._convert_float
calls = []


def record_conversion(data, src, dst, mode):
    value = original(data, src, dst, mode)
    if src == tl.float32 and dst == tl.bfloat16:
        correct = bf16_bits(data)
        calls.append({"rounding_mode": str(mode), "elements": int(data.size),
                      "different_from_torch_rtne": int(np.count_nonzero(value != correct))})
    return value


interpreter._convert_float = record_conversion
try:
    actual = kernel_module.residual_rmsnorm_forward(*inputs, reference.eps)
finally:
    interpreter._convert_float = original


def controlled_rtne(data, src, dst, mode):
    if src == tl.float32 and dst == tl.bfloat16:
        return bf16_bits(data)
    return original(data, src, dst, mode)


interpreter._convert_float = controlled_rtne
try:
    corrected = kernel_module.residual_rmsnorm_forward(*inputs, reference.eps)
finally:
    interpreter._convert_float = original
probe = np.array([1.00390625, 1.01171875, -1.00390625, -1.01171875], dtype=np.float32)
probe_original = original(probe, tl.float32, tl.bfloat16, interpreter._ir.ROUNDING_MODE.RTNE)
probe_expected = bf16_bits(probe)
report = {
    "scope": "one frozen seed-0 CPU input; controlled interpreter-only intervention; no GPU/reward claim",
    "torch": torch.__version__, "triton": triton.__version__, "shape": list(inputs[0].shape),
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            root / "tasks/residual_rmsnorm/kernels/triton/residual_rmsnorm.py",
            Path(interpreter.__file__))},
    "original": {"normalized": comparison(actual[0], expected[0]),
                 "residual_sum": comparison(actual[1], expected[1])},
    "controlled_rtne_only": {"normalized": comparison(corrected[0], expected[0]),
                           "residual_sum": comparison(corrected[1], expected[1])},
    "executed_float32_to_bf16_conversions": calls,
    "rounding_tie_probe": {"float32": probe.tolist(), "interpreter_bf16_bits": probe_original.tolist(),
                          "torch_rtne_bf16_bits": probe_expected.tolist()},
    "production_changed": False,
}
out = root / "artifacts" / ("task-residual-rounding-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
out.mkdir(exist_ok=False)
(out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({key: report[key] for key in ("shape", "original", "controlled_rtne_only", "rounding_tie_probe")}))
print(out / "report.json")
