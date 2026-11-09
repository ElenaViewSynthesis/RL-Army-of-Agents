import os

import torch
import triton
import triton.language as tl


@triton.jit
def masked_mean_kernel(x_ptr, lengths_ptr, out_ptr, l, BLOCK_L: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_L)
    length = tl.load(lengths_ptr + row)
    mask = offs < length
    x = tl.load(x_ptr + row * l + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(x) / tl.maximum(length.to(tl.float32), 1.0))


def masked_mean_forward(x, lengths):
    if x.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
        raise ValueError("masked_mean requires x [B,L] and lengths [B]")
    if x.device != lengths.device:
        raise ValueError("x and lengths must have the same device")
    if lengths.dtype != torch.int32:
        raise ValueError("masked_mean requires int32 lengths")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("masked_mean requires float32 or bfloat16 input")
    if not x.is_contiguous() or not lengths.is_contiguous():
        raise ValueError("masked_mean requires contiguous inputs")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, l = x.shape
    out = torch.empty(b, dtype=torch.float32, device=x.device)
    if b:
        masked_mean_kernel[(b,)](x, lengths, out, l,
                                 BLOCK_L=triton.next_power_of_2(max(l, 1)))
    return out
