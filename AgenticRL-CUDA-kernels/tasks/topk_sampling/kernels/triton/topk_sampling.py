import os

import torch
import triton
import triton.language as tl


@triton.jit
def _sortable_key(x, idx):
    u = x.to(tl.uint32, bitcast=True)
    u = tl.where(x == 0.0, tl.zeros_like(u), u)
    ordered = tl.where(u >= 0x80000000, 0xFFFFFFFF - u, u + 0x80000000)
    return (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - idx.to(tl.uint64))


@triton.jit
def topk_chunk_kernel(x_ptr, scratch_ptr, v, k, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < v
    x = tl.load(x_ptr + row * v + offs, mask=mask, other=float("-inf"))
    keys = tl.where(mask, _sortable_key(x, offs), tl.zeros_like(_sortable_key(x, offs)))
    keys = tl.sort(keys, descending=True)
    j = tl.arange(0, CHUNK)
    tl.store(scratch_ptr + (row * tl.num_programs(1) + chunk) * k + j,
             keys.to(tl.int64, bitcast=True), mask=j < k)


@triton.jit
def topk_final_kernel(x_ptr, u_ptr, scratch_ptr, out_ptr, v, k, chunks, top_p,
                      CAND: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, CAND)
    keys = tl.load(scratch_ptr + row * chunks * k + offs,
                   mask=offs < chunks * k, other=0).to(tl.uint64, bitcast=True)
    keys = tl.sort(keys, descending=True)
    top = tl.where(offs < k, keys, tl.zeros_like(keys))
    token_ids = (0xFFFFFFFF - (top & 0xFFFFFFFF)).to(tl.int32)
    logits = tl.load(x_ptr + row * v + token_ids, mask=offs < k, other=float("-inf"))
    mx = tl.max(tl.where(offs < k, logits, float("-inf")))
    ex = tl.exp(logits - mx)
    ex = tl.where(offs < k, ex, 0.0)
    probs = ex / tl.sum(ex)
    cum = tl.cumsum(probs)
    keep = (cum - probs) < top_p
    filt = tl.where(keep, probs, 0.0)
    cdf = tl.cumsum(filt / tl.sum(filt))
    uniform = tl.load(u_ptr + row)
    rank = tl.sum(((cdf <= uniform) & (offs < k)).to(tl.int32))
    rank = tl.minimum(rank, k - 1)
    chosen = tl.sum(tl.where(offs == rank, token_ids, 0))
    tl.store(out_ptr + row, chosen)


def topk_sampling_forward(logits, uniforms, top_k, top_p):
    if logits.ndim != 2 or uniforms.ndim != 1 or uniforms.shape[0] != logits.shape[0]:
        raise ValueError("topk_sampling requires logits [B,V] and uniforms [B]")
    if logits.dtype != torch.float32 or uniforms.dtype != torch.float32:
        raise ValueError("topk_sampling requires float32 inputs")
    if logits.device != uniforms.device:
        raise ValueError("inputs must share a device")
    if not logits.is_contiguous() or not uniforms.is_contiguous():
        raise ValueError("topk_sampling requires contiguous inputs")
    if logits.shape[1] <= 0 or not (1 <= top_k <= min(32, logits.shape[1])):
        raise ValueError("top_k must be in [1,min(32,V)] and V>0")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0,1]")
    if logits.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, v = logits.shape
    chunks = triton.cdiv(v, 1024)
    scratch = torch.empty((b, chunks * top_k), dtype=torch.int64, device=logits.device)
    out = torch.empty(b, dtype=torch.int32, device=logits.device)
    if b and v:
        topk_chunk_kernel[(b, chunks)](logits, scratch, v, top_k, CHUNK=1024, num_warps=4)
        topk_final_kernel[(b,)](logits, uniforms, scratch, out, v, top_k, chunks,
                                float(top_p), CAND=triton.next_power_of_2(chunks * top_k),
                                num_warps=8)
    return out
