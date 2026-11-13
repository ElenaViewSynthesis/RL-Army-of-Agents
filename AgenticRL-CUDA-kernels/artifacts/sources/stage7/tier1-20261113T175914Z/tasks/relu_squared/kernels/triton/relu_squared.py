import os

import torch
import triton
import triton.language as tl


@triton.jit
def relu_squared_triton_chunks(x_ptr, out_ptr, h, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < h
    value = tl.load(x_ptr + row * h + cols, mask=mask, other=0.0).to(tl.float32)
    positive = tl.maximum(value, 0.0)
    tl.store(out_ptr + row * h + cols, (positive * positive).to(out_ptr.dtype.element_ty), mask=mask)


def relu_squared_forward(x):
    if x.ndim != 2:
        raise ValueError("relu_squared requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("relu_squared requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("relu_squared requires a contiguous input")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    rows, h = x.shape
    out = torch.empty_like(x)
    if rows and h:
        relu_squared_triton_chunks[(rows, triton.cdiv(h, 1024))](x, out, h, BLOCK=1024, num_warps=4)
    return out
