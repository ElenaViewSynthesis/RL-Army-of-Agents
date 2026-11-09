import os

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, scale, s, d,
                           causal: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    base = pid_bh * s * d
    q = tl.load(q_ptr + base + offs_m[:, None] * d + offs_d[None, :],
                mask=(offs_m[:, None] < s) & (offs_d[None, :] < d), other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    hi = (pid_m + 1) * BLOCK_M if causal else s
    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kt = tl.load(k_ptr + base + offs_n[:, None] * d + offs_d[None, :],
                     mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), other=0.0)
        scores = tl.dot(q, tl.trans(kt)) * scale
        scores = tl.where(offs_n[None, :] < s, scores, float("-inf"))
        if causal:
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        p = tl.exp(scores - m_new[:, None])
        p = tl.where((offs_n[None, :] < s) & (p == p), p, 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        vt = tl.load(v_ptr + base + offs_n[:, None] * d + offs_d[None, :],
                     mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_ptr.dtype.element_ty), vt)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(out_ptr + base + offs_m[:, None] * d + offs_d[None, :],
             acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < s) & (offs_d[None, :] < d))


def flash_attention_forward(q, k, v, causal):
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("flash_attention requires q,k,v [B,H,S,D] of equal shape")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("flash_attention requires bfloat16 inputs")
    if q.shape[3] != 64:
        raise ValueError("flash_attention requires head dim D=64")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("flash_attention requires contiguous inputs")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must share a device")
    if q.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, h, s, d = q.shape
    out = torch.empty_like(q)
    if b * h * s * d:
        flash_attention_kernel[(triton.cdiv(s, 64), b * h)](
            q, k, v, out, 1.0 / (d ** 0.5), s, d,
            causal=bool(causal), BLOCK_M=64, BLOCK_N=64, BLOCK_D=64,
            num_warps=4, num_stages=3)
    return out
