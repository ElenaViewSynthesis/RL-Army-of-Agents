import os

import torch
import triton
import triton.language as tl

CHUNK = 4096


@triton.jit
def log_softmax_chunk_kernel(x_ptr, pmax_ptr, psum_ptr, v, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < v
    x = tl.load(x_ptr + row * v + offs, mask=mask, other=float("-inf")).to(tl.float32)
    pm = tl.max(x)
    ps = tl.sum(tl.exp(x - pm))
    tl.store(pmax_ptr + row * tl.num_programs(1) + chunk, pm)
    tl.store(psum_ptr + row * tl.num_programs(1) + chunk, ps)


@triton.jit
def log_softmax_gather_kernel(x_ptr, ids_ptr, pmax_ptr, psum_ptr, out_ptr, v, chunks,
                              BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < chunks
    pm = tl.load(pmax_ptr + row * chunks + offs, mask=mask, other=float("-inf"))
    ps = tl.load(psum_ptr + row * chunks + offs, mask=mask, other=0.0)
    m = tl.max(pm)
    s = tl.sum(ps * tl.exp(pm - m))
    target = tl.load(ids_ptr + row)
    xt = tl.load(x_ptr + row * v + target).to(tl.float32)
    tl.store(out_ptr + row, xt - m - tl.log(s))


def log_softmax_gather_forward(logits, token_ids):
    if logits.ndim != 2 or token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("log_softmax_gather requires logits [B,V] and token_ids [B]")
    if token_ids.dtype != torch.int64:
        raise ValueError("log_softmax_gather requires int64 token_ids")
    if logits.device != token_ids.device:
        raise ValueError("logits and token_ids must have the same device")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("log_softmax_gather requires float32 or bfloat16 input")
    if not logits.is_contiguous() or not token_ids.is_contiguous():
        raise ValueError("log_softmax_gather requires contiguous inputs")
    if logits.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, v = logits.shape
    out = torch.empty(b, dtype=torch.float32, device=logits.device)
    if b and v:
        chunks = triton.cdiv(v, CHUNK)
        pmax = torch.empty((b, chunks), dtype=torch.float32, device=logits.device)
        psum = torch.empty((b, chunks), dtype=torch.float32, device=logits.device)
        log_softmax_chunk_kernel[(b, chunks)](logits, pmax, psum, v, CHUNK=CHUNK)
        log_softmax_gather_kernel[(b,)](logits, token_ids, pmax, psum, out, v, chunks,
                                        BLOCK_C=triton.next_power_of_2(chunks))
    return out
