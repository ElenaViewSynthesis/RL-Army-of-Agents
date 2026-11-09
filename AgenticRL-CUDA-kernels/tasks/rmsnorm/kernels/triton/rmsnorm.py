import os

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, h, eps, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < h
    x = tl.load(x_ptr + row * h + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x) / h + eps)
    tl.store(out_ptr + row * h + offs, (x * inv * w).to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm_forward(x, weight, eps):
    if x.ndim != 2 or weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("rmsnorm requires x [B,H] and weight [H]")
    if x.device != weight.device:
        raise ValueError("x and weight must have the same device")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("rmsnorm requires float32 or bfloat16 input")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("rmsnorm requires contiguous inputs")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, h = x.shape
    out = torch.empty_like(x)
    if b and h:
        rmsnorm_kernel[(b,)](x, weight, out, h, float(eps),
                             BLOCK_H=triton.next_power_of_2(h),
                             num_warps=8 if h > 1024 else 4)
    return out
