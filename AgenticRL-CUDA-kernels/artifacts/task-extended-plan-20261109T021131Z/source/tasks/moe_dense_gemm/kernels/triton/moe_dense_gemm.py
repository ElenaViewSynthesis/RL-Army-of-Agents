import os

import torch
import triton
import triton.language as tl


@triton.jit
def moe_fc1_kernel(x_ptr, gate_ptr, up_ptr, out_ptr, t, h, e, i,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    expert = rn // (2 * i)
    which = (rn // i) % 2
    inner = rn % i
    for k0 in range(0, h, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * h + rk[None, :],
                    mask=(rm[:, None] < t) & (rk[None, :] < h), other=0.0)
        off = expert[None, :] * h * i + rk[:, None] * i + inner[None, :]
        is_gate = which[None, :] == 0
        wmask = (rk[:, None] < h) & (rn[None, :] < 2 * e * i)
        g = tl.load(gate_ptr + off, mask=wmask & is_gate, other=0.0)
        u = tl.load(up_ptr + off, mask=wmask & ~is_gate, other=0.0)
        w = (g + u).to(x_ptr.dtype.element_ty)
        acc = tl.dot(x, w, acc)
    tl.store(out_ptr + rm[:, None] * (2 * e * i) + rn[None, :], acc,
             mask=(rm[:, None] < t) & (rn[None, :] < 2 * e * i))


@triton.jit
def moe_act_kernel(fc1_ptr, alpha_ptr, out_ptr, t, e, i, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = t * e * i
    mask = offs < total
    ii = offs % i
    e_ = (offs // i) % e
    tok = offs // (i * e)
    g = tl.load(fc1_ptr + (tok * e + e_) * 2 * i + ii, mask=mask, other=0.0)
    u = tl.load(fc1_ptr + (tok * e + e_) * 2 * i + i + ii, mask=mask, other=0.0)
    a = tl.load(alpha_ptr + tok * e + e_, mask=mask, other=0.0)
    act = g * tl.sigmoid(g) * u * a
    tl.store(out_ptr + offs, act.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def moe_down_kernel(act_ptr, down_ptr, out_ptr, t, h, ei,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, ei, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(act_ptr + rm[:, None] * ei + rk[None, :],
                    mask=(rm[:, None] < t) & (rk[None, :] < ei), other=0.0)
        w = tl.load(down_ptr + rk[:, None] * h + rn[None, :],
                    mask=(rk[:, None] < ei) & (rn[None, :] < h), other=0.0)
        acc = tl.dot(a, w, acc)
    tl.store(out_ptr + rm[:, None] * h + rn[None, :], acc,
             mask=(rm[:, None] < t) & (rn[None, :] < h))


def moe_dense_gemm_forward(x, gate, up, down, alpha):
    if x.ndim != 2 or gate.ndim != 3 or up.shape != gate.shape or down.ndim != 3:
        raise ValueError("moe_dense_gemm requires x [T,H], gate/up [E,H,I], down [E,I,H]")
    e, h, i = gate.shape
    if x.shape[1] != h or down.shape != (e, i, h) or alpha.shape != (x.shape[0], e):
        raise ValueError("moe_dense_gemm shape mismatch")
    if not all(t.dtype == torch.bfloat16 for t in (x, gate, up, down)) or alpha.dtype != torch.float32:
        raise ValueError("moe_dense_gemm requires bfloat16 weights and float32 alpha")
    dev = x.device
    if not all(t.device == dev for t in (gate, up, down, alpha)):
        raise ValueError("inputs must share a device")
    if not all(t.is_contiguous() for t in (x, gate, up, down, alpha)):
        raise ValueError("moe_dense_gemm requires contiguous inputs")
    if dev.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    t = x.shape[0]
    fc1 = torch.empty((t, 2 * e * i), dtype=torch.float32, device=dev)
    act = torch.empty((t, e * i), dtype=torch.bfloat16, device=dev)
    out = torch.empty((t, h), dtype=torch.float32, device=dev)
    if t and e * i:
        moe_fc1_kernel[(triton.cdiv(t, 32), triton.cdiv(2 * e * i, 64))](
            x, gate, up, fc1, t, h, e, i, BM=32, BN=64, BK=32, num_warps=4, num_stages=3)
        moe_act_kernel[(triton.cdiv(t * e * i, 1024),)](fc1, alpha, act, t, e, i, BLOCK=1024)
        moe_down_kernel[(triton.cdiv(t, 32), triton.cdiv(h, 64))](
            act, down.reshape(e * i, h), out, t, h, e * i,
            BM=32, BN=64, BK=32, num_warps=4, num_stages=3)
    return out
