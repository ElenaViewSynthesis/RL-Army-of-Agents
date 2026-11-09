import os

import torch
import triton
import triton.language as tl


@triton.jit
def prefix_scan_block_kernel(x_ptr, out_ptr, totals_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(x_ptr + offs, mask=offs < n, other=0)
    tl.store(totals_ptr + pid, tl.sum(vals))
    vals = tl.cumsum(vals)
    tl.store(out_ptr + offs, vals, mask=offs < n)


@triton.jit
def prefix_scan_offset_kernel(out_ptr, totals_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid > 0:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        base = tl.load(totals_ptr + pid - 1)
        vals = tl.load(out_ptr + offs, mask=offs < n, other=0)
        tl.store(out_ptr + offs, vals + base, mask=offs < n)


def _scan(x):
    n = x.numel()
    out = torch.empty_like(x)
    blocks = triton.cdiv(n, 1024)
    totals = torch.empty(blocks, dtype=x.dtype, device=x.device)
    prefix_scan_block_kernel[(blocks,)](x, out, totals, n, BLOCK=1024)
    if blocks > 1:
        scanned_totals = _scan(totals)
        prefix_scan_offset_kernel[(blocks,)](out, scanned_totals, n, BLOCK=1024)
    return out


def prefix_sum_forward(x):
    if x.ndim != 1:
        raise ValueError("prefix_sum requires a 1D input")
    if x.dtype not in (torch.int32, torch.float32):
        raise ValueError("prefix_sum requires int32 or float32 input")
    if not x.is_contiguous():
        raise ValueError("prefix_sum requires contiguous input")
    if x.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    if not x.numel():
        return torch.empty_like(x)
    return _scan(x)
