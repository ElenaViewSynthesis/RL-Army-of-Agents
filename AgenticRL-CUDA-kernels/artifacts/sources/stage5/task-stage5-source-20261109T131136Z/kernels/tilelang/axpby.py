"""Fused alpha * a + b in TileLang. GPU only.

TileLang's release wheels have no CPU interpreter (its LLVM CPU target needs a
source build), so this kernel runs on CUDA devices only. TileLang names the
device symbol after the prim_func (`axpby_tilelang` -> `axpby_tilelang_kernel`),
which is how the provenance profiler credits it. Verified on an H200 (Modal run
20261106T185236Z, TileLang 0.1.15): correct output, observed symbol
`axpby_tilelang_kernel`, provenance 1.0.
"""

import functools

import tilelang
import tilelang.language as T
import torch

_DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16",
           torch.float32: "float32", torch.float64: "float64"}
BLOCK, THREADS = 1024, 256


def _program(n, alpha, dtype, block, threads):
    @T.prim_func
    def axpby_tilelang(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype),
                       C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, block), threads=threads) as bx:
            for i in T.Parallel(block):
                # TileLang inserts the tail bounds check for n % block != 0.
                C[bx * block + i] = A[bx * block + i] * T.Cast(dtype, alpha) + B[bx * block + i]

    return axpby_tilelang


@functools.lru_cache(maxsize=None)
def _compiled(n, alpha, dtype):
    # Pin the default CUDA target; experimental targets (cutedsl, llvm) are out of scope.
    return tilelang.compile(_program(n, alpha, dtype, BLOCK, THREADS), out_idx=-1, target="cuda")


def axpby_forward(a, b, alpha):
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same shape, dtype, and device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("axpby requires contiguous inputs")
    if a.dtype not in _DTYPES:
        raise ValueError("axpby requires floating point inputs")
    if a.device.type != "cuda":
        raise ValueError("TileLang axpby requires CUDA tensors (no CPU interpreter)")
    n = a.numel()
    if n == 0:
        return torch.empty_like(a)
    out = _compiled(n, float(alpha), _DTYPES[a.dtype])(a.view(-1), b.view(-1))
    return out.view(a.shape)
