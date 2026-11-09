import os

import torch
import triton
import triton.language as tl


@triton.jit
def _group_tile(a_ptr, b_ptr, o_ptr, pid_m, pid_n, m, n, k,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * k + rk[None, :],
                    mask=(rm[:, None] < m) & (rk[None, :] < k), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * n + rn[None, :],
                    mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(o_ptr + rm[:, None] * n + rn[None, :], acc,
             mask=(rm[:, None] < m) & (rn[None, :] < n))


@triton.jit
def grouped_gemm_kernel(a0, a1, a2, a3, a4, a5, a6, a7,
                        b0, b1, b2, b3, b4, b5, b6, b7,
                        o0, o1, o2, o3, o4, o5, o6, o7,
                        m0, m1, m2, m3, m4, m5, m6, m7, n, k,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)
    if g == 0:
        _group_tile(a0, b0, o0, pid_m, pid_n, m0, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    elif g == 1:
        _group_tile(a1, b1, o1, pid_m, pid_n, m1, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    elif g == 2:
        _group_tile(a2, b2, o2, pid_m, pid_n, m2, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    elif g == 3:
        _group_tile(a3, b3, o3, pid_m, pid_n, m3, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    elif g == 4:
        _group_tile(a4, b4, o4, pid_m, pid_n, m4, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    elif g == 5:
        _group_tile(a5, b5, o5, pid_m, pid_n, m5, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    elif g == 6:
        _group_tile(a6, b6, o6, pid_m, pid_n, m6, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    else:
        _group_tile(a7, b7, o7, pid_m, pid_n, m7, n, k, BLOCK_M, BLOCK_N, BLOCK_K)


def grouped_gemm_forward(matrices_a, matrices_b):
    if len(matrices_a) != len(matrices_b) or not 1 <= len(matrices_a) <= 8:
        raise ValueError("grouped_gemm requires 1-8 paired matrices")
    g = len(matrices_a)
    for a, b in zip(matrices_a, matrices_b):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("grouped_gemm requires a [M,K] and b [K,N] per group")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError("grouped_gemm requires bfloat16 inputs")
        if not a.is_contiguous() or not b.is_contiguous():
            raise ValueError("grouped_gemm requires contiguous inputs")
    k = matrices_a[0].shape[1]
    n = matrices_b[0].shape[1]
    if any(a.shape[1] != k for a in matrices_a) or any(b.shape != (k, n) for b in matrices_b):
        raise ValueError("grouped_gemm requires shared K and N across groups")
    dev = matrices_a[0].device
    if any(x.device != dev for x in (*matrices_a, *matrices_b)):
        raise ValueError("grouped_gemm inputs must share a device")
    if dev.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    outs = [torch.empty((a.shape[0], n), dtype=torch.float32, device=dev) for a in matrices_a]
    max_m = max(a.shape[0] for a in matrices_a)
    if max_m and n:
        pad = 8 - g
        ms = [a.shape[0] for a in matrices_a] + [0] * pad
        grouped_gemm_kernel[(triton.cdiv(max_m, 32), triton.cdiv(n, 64), g)](
            *matrices_a, *([matrices_a[0]] * pad), *matrices_b, *([matrices_b[0]] * pad),
            *outs, *([outs[0]] * pad), *ms, n, k,
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3)
    return outs
