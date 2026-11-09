import os

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_residual_gelu_kernel(x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr,
                                   h, eps, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < h
    x = tl.load(x_ptr + row * h + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x) / h
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered) / h
    norm = centered * tl.rsqrt(var + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * h + offs, mask=mask, other=0.0).to(tl.float32)
    value = norm * w + b + r
    out = 0.5 * value * (1.0 + tl.math.erf(value * 0.7071067811865476))
    tl.store(out_ptr + row * h + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def layernorm_residual_gelu_forward(x, residual, weight, bias, eps):
    if x.ndim != 2:
        raise ValueError("layernorm_residual_gelu requires x [B,H]")
    for name, t in (("residual", residual), ("weight", weight), ("bias", bias)):
        if t.device != x.device:
            raise ValueError(f"{name} must share x's device")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if residual.shape != x.shape or weight.shape != (x.shape[1],) or bias.shape != (x.shape[1],):
        raise ValueError("shape mismatch against x [B,H], weight/bias [H]")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("layernorm_residual_gelu requires float32 or bfloat16 input")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, h = x.shape
    out = torch.empty_like(x)
    if b and h:
        layernorm_residual_gelu_kernel[(b,)](x, residual, weight, bias, out, h, float(eps),
                                             BLOCK_H=triton.next_power_of_2(h),
                                             num_warps=8 if h > 1024 else 4)
    return out
