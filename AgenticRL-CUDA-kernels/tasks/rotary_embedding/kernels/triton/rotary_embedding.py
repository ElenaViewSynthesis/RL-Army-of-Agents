import os

import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_triton_rows(x_ptr, cos_ptr, sin_ptr, out_ptr, n_rows, seq,
                                 HALF: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, HALF)
    mask = (rows < n_rows)[:, None]
    base = rows[:, None] * (2 * HALF) + cols[None, :]
    table = (rows % seq)[:, None] * HALF + cols[None, :]
    x1 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + table, mask=mask, other=0.0)
    s = tl.load(sin_ptr + table, mask=mask, other=0.0)
    tl.store(out_ptr + base, (x1 * c - x2 * s).to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + base + HALF, (x2 * c + x1 * s).to(out_ptr.dtype.element_ty), mask=mask)


def _check(q, k, cos, sin):
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("rotary_embedding requires q [B,Hq,S,D] and k [B,Hkv,S,D]")
    s, d = q.shape[2], q.shape[3]
    if d % 2 or cos.shape != (s, d // 2) or sin.shape != (s, d // 2):
        raise ValueError("rotary_embedding requires even D and cos/sin [S,D/2]")
    if q.dtype not in (torch.bfloat16, torch.float32) or k.dtype != q.dtype:
        raise ValueError("rotary_embedding requires float32 or bfloat16 q and k")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("rotary_embedding requires float32 cos/sin tables")
    if not all(t.is_contiguous() for t in (q, k, cos, sin)):
        raise ValueError("rotary_embedding requires contiguous inputs")


def rotary_embedding_forward(q, k, cos, sin):
    _check(q, k, cos, sin)
    if q.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    seq, half = q.shape[2], q.shape[3] // 2
    if half & (half - 1):
        raise ValueError("rotary_embedding requires D/2 to be a power of two")
    outputs = []
    for x in (q, k):
        out = torch.empty_like(x)
        n_rows = x.numel() // (2 * half)
        if n_rows:
            rotary_embedding_triton_rows[(triton.cdiv(n_rows, 16),)](
                x, cos, sin, out, n_rows, seq, HALF=half, ROWS=16, num_warps=4)
        outputs.append(out)
    return outputs[0], outputs[1]
