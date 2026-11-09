import os

import torch
import triton
import triton.language as tl


@triton.jit
def packed_varlen_reduction_kernel(v_ptr, off_ptr, out_ptr, is_mean, BLOCK: tl.constexpr):
    seg = tl.program_id(0)
    start = tl.load(off_ptr + seg)
    end = tl.load(off_ptr + seg + 1)
    length = end - start
    acc = 0.0
    for i0 in range(0, length, BLOCK):
        offs = start + i0 + tl.arange(0, BLOCK)
        vals = tl.load(v_ptr + offs, mask=offs < end, other=0.0)
        acc += tl.sum(vals)
    if is_mean:
        acc = acc / tl.maximum(length.to(tl.float32), 1.0)
    tl.store(out_ptr + seg, acc)


def packed_varlen_reduction_forward(values, offsets, reduction):
    if values.ndim != 1 or offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError("packed_varlen_reduction requires values [N] and offsets [B+1]")
    if values.dtype != torch.float32:
        raise ValueError("values must be float32")
    if offsets.dtype != torch.int32:
        raise ValueError("offsets must be int32")
    if values.device != offsets.device:
        raise ValueError("values and offsets must share a device")
    if not values.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("packed_varlen_reduction requires contiguous inputs")
    if values.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b = offsets.shape[0] - 1
    out = torch.empty(b, dtype=torch.float32, device=values.device)
    if b:
        packed_varlen_reduction_kernel[(b,)](values, offsets, out,
                                             1 if reduction == "mean" else 0, BLOCK=1024)
    return out
