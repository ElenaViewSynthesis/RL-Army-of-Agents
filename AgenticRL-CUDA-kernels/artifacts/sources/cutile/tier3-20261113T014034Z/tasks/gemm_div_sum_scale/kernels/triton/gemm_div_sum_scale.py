import os

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_div_sum_matmul_kernel(a_ptr, b_ptr, product_ptr, m, n, k,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * k + rk[None, :],
                    mask=(rm[:, None] < m) & (rk[None, :] < k), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * n + rn[None, :],
                    mask=(rk[:, None] < k) & (rn[None, :] < n), other=0.0)
        acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
    tl.store(product_ptr + rm[:, None] * n + rn[None, :], acc,
             mask=(rm[:, None] < m) & (rn[None, :] < n))


@triton.jit
def gemm_div_sum_rowsum_kernel(product_ptr, out_ptr, m, n, divisor, scale,
                               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for n0 in range(0, n, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        vals = tl.load(product_ptr + row * n + offs, mask=offs < n, other=0.0)
        acc += tl.sum(vals)
    tl.store(out_ptr + row, (acc / divisor) * scale)


def gemm_div_sum_scale_forward(a, b, divisor, scale):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm_div_sum_scale requires a [M,K] and b [K,N]")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same dtype and device")
    if a.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("gemm_div_sum_scale requires float32 or bfloat16 inputs")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("gemm_div_sum_scale requires contiguous inputs")
    if a.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    product = torch.empty((m, n), dtype=torch.float32, device=a.device)
    out = torch.empty((m, 1), dtype=torch.float32, device=a.device)
    if m and n:
        gemm_div_sum_matmul_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            a, b, product, m, n, k, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
        gemm_div_sum_rowsum_kernel[(m,)](product, out, m, n,
                                         float(divisor), float(scale), BLOCK_N=1024)
    return out
