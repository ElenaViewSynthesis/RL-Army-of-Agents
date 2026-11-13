import os

import torch
import triton
import triton.language as tl


@triton.jit
def residual_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, out_ptr, z_ptr, h, eps,
                            BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < h
    x = tl.load(x_ptr + row * h + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * h + offs, mask=mask, other=0.0).to(tl.float32)
    z = (x + r).to(z_ptr.dtype.element_ty)
    tl.store(z_ptr + row * h + offs, z, mask=mask)
    v = z.to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v) / h + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * h + offs, (v * inv * w).to(out_ptr.dtype.element_ty), mask=mask)


def residual_rmsnorm_forward(x, residual, weight, eps):
    if x.ndim != 2 or residual.shape != x.shape or weight.shape != (x.shape[1],):
        raise ValueError("residual_rmsnorm requires x,residual [B,H] and weight [H]")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("residual_rmsnorm requires bfloat16 inputs")
    if not (x.device == residual.device == weight.device):
        raise ValueError("inputs must share a device")
    if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("residual_rmsnorm requires contiguous inputs")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, h = x.shape
    out = torch.empty_like(x)
    z = torch.empty_like(x)
    if b and h:
        residual_rmsnorm_kernel[(b,)](x, residual, weight, out, z, h, float(eps),
                                      BLOCK_H=triton.next_power_of_2(h),
                                      num_warps=8 if h > 1024 else 4)
    return out, z
