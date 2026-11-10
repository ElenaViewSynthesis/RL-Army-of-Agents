"""Fused alpha * a + b. CPU interpretation pins one config (no GPU timer)."""

import os

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": size}, num_warps=4) for size in
             ((128,) if os.environ.get("TRITON_INTERPRET") == "1" else (128, 256, 512))],
    key=["n_elements"],
)
@triton.jit
def axpby_kernel(a_ptr, b_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, alpha * a + b, mask=mask)


def axpby_forward(a, b, alpha):
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same shape, dtype, and device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("axpby requires contiguous inputs")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("axpby requires floating point inputs")
    if a.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    out = torch.empty_like(a)
    n = a.numel()
    if n:
        axpby_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)](a, b, out, alpha, n)
    return out
