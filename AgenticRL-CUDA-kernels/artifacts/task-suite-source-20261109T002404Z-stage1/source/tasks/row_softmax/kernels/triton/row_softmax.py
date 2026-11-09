import os

import torch
import triton
import triton.language as tl


@triton.jit
def row_softmax_kernel(x_ptr, out_ptr, h, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < h
    x = tl.load(x_ptr + row * h + offs, mask=mask, other=float("-inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x))
    p = tl.where(mask, p, 0.0)
    tl.store(out_ptr + row * h + offs, (p / tl.sum(p)).to(out_ptr.dtype.element_ty), mask=mask)


def row_softmax_forward(x):
    if x.ndim != 2:
        raise ValueError("row_softmax requires a 2D input")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("row_softmax requires float16, bfloat16 or float32 input")
    if not x.is_contiguous():
        raise ValueError("row_softmax requires contiguous input")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, h = x.shape
    out = torch.empty_like(x)
    if b and h:
        row_softmax_kernel[(b,)](x, out, h, BLOCK_H=triton.next_power_of_2(h),
                                 num_warps=8 if h > 1024 else 4)
    return out
