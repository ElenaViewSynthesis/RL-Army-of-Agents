import os

import torch
import triton
import triton.language as tl

_CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                          num_warps=w, num_stages=s)
            for bm, bn, bk, w, s in ((64, 128, 32, 4, 3), (128, 128, 32, 4, 4),
                                     (128, 256, 64, 8, 3), (128, 128, 64, 4, 4),
                                     (64, 256, 64, 4, 4), (32, 128, 64, 4, 4))]


@triton.autotune(configs=_CONFIGS[:1] if os.environ.get("TRITON_INTERPRET") == "1" else _CONFIGS,
                 key=["m", "n", "k"])
@triton.jit
def gemm_tensor_core_kernel(a_ptr, b_ptr, bias_ptr, out_ptr, m, n, k,
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
        acc = tl.dot(a, b, acc)
    bias = tl.load(bias_ptr + rn, mask=rn < n, other=0.0).to(tl.float32)
    tl.store(out_ptr + rm[:, None] * n + rn[None, :], acc + bias[None, :],
             mask=(rm[:, None] < m) & (rn[None, :] < n))


def gemm_tensor_core_forward(a, b, bias):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm_tensor_core requires a [M,K] and b [K,N]")
    if a.dtype != b.dtype or a.device != b.device or a.device != bias.device:
        raise ValueError("a, b and bias must share dtype and device")
    if bias.ndim != 1 or bias.shape[0] != b.shape[1] or bias.dtype != torch.float32:
        raise ValueError("bias must be float32 [N]")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("gemm_tensor_core requires float16 or bfloat16 inputs")
    if not a.is_contiguous() or not b.is_contiguous() or not bias.is_contiguous():
        raise ValueError("gemm_tensor_core requires contiguous inputs")
    if a.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    if m and n:
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
        gemm_tensor_core_kernel[grid](a, b, bias, out, m, n, k)
    return out
