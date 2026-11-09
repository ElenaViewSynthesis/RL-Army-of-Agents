import os

import torch
import triton
import triton.language as tl


@triton.jit
def decode_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, scale, s, d,
                            kv_heads: tl.constexpr, repeats: tl.constexpr,
                            BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    qh = tl.program_id(0)
    b = tl.program_id(1)
    kvh = qh // repeats
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(q_ptr + (b * tl.num_programs(0) + qh) * d + offs_d,
                mask=offs_d < d, other=0.0).to(tl.float32)
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    kv_base = (b * kv_heads + kvh) * s * d
    for n0 in range(0, s, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + kv_base + offs_n[:, None] * d + offs_d[None, :],
                     mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), other=0.0).to(tl.float32)
        scores = tl.sum(kt * q[None, :], 1) * scale
        scores = tl.where(offs_n < s, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores))
        p = tl.where(offs_n < s, tl.exp(scores - m_new), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p)
        vt = tl.load(v_ptr + kv_base + offs_n[:, None] * d + offs_d[None, :],
                     mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vt, 0)
        m_i = m_new
    acc = acc / l_i
    tl.store(out_ptr + (b * tl.num_programs(0) + qh) * d + offs_d,
             acc.to(out_ptr.dtype.element_ty), mask=offs_d < d)


def decode_attention_forward(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("decode_attention requires q [B,Hq,1,D] and k,v [B,Hkv,S,D]")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3] or q.shape[2] != 1:
        raise ValueError("decode_attention shape mismatch")
    if k.shape[1] <= 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("query heads must divide evenly over kv heads")
    if q.shape[3] != 64:
        raise ValueError("decode_attention requires head dim D=64")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("decode_attention requires bfloat16 inputs")
    if not (q.device == k.device == v.device):
        raise ValueError("inputs must share a device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("decode_attention requires contiguous inputs")
    if q.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, hq, _, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    out = torch.empty_like(q)
    if b * hq * s * d:
        decode_attention_kernel[(hq, b)](q, k, v, out, 1.0 / (d ** 0.5), s, d,
                                         kv_heads=hkv, repeats=hq // hkv,
                                         BLOCK_N=128, BLOCK_D=64, num_warps=4)
    return out
