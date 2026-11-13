import os

import torch
import triton
import triton.language as tl


@triton.jit
def geglu_tanh_triton_chunks(x_ptr, out_ptr, h, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < h
    a = tl.load(x_ptr + row * 2 * h + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row * 2 * h + h + cols, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (b + 0.044715 * b * b * b)
    tanh = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    tl.store(out_ptr + row * h + cols, (a * 0.5 * b * (1.0 + tanh)).to(out_ptr.dtype.element_ty), mask=mask)


def geglu_tanh_forward(x):
    if x.ndim != 2:
        raise ValueError("geglu_tanh requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("geglu_tanh requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("geglu_tanh requires a contiguous input")
    if x.shape[1] % 2:
        raise ValueError("geglu_tanh requires an even last dimension [B,2H]")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    rows, h = x.shape[0], x.shape[1] // 2
    out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    if rows and h:
        geglu_tanh_triton_chunks[(rows, triton.cdiv(h, 1024))](x, out, h, BLOCK=1024, num_warps=4)
    return out
