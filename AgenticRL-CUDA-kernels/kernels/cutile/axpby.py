"""Fused alpha * a + b in cuTile (NVIDIA CUDA Tile, `cuda.tile`). GPU only.

Needs compute capability 8.x-12.x, driver r580+ and Python 3.11+. cuTile has no
documented CPU interpreter. Kernels compile on first launch, which the harness
compile stage triggers. Not yet run on hardware here.
"""

import math

import cuda.tile as ct
import torch

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
MAX_TILE = 1024


@ct.kernel
def axpby_cutile_kernel(a, b, c, alpha, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    # gather zero-fills and scatter skips out-of-bounds indices: this covers the tail.
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)
    ct.scatter(c, indices, (alpha * a_tile + b_tile).astype(c.dtype))


def axpby_forward(a, b, alpha):
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same shape, dtype, and device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("axpby requires contiguous inputs")
    if a.dtype not in _DTYPES:
        raise ValueError("axpby requires floating point inputs")
    if a.device.type != "cuda":
        raise ValueError("cuTile axpby requires CUDA tensors (no CPU interpreter)")
    out = torch.empty_like(a)
    n = a.numel()
    if n:
        tile = min(MAX_TILE, 2 ** math.ceil(math.log2(n)))
        grid = (math.ceil(n / tile), 1, 1)
        # Launch on the caller's current stream (stream/device contract).
        ct.launch(torch.cuda.current_stream(), grid, axpby_cutile_kernel,
                  (a.view(-1), b.view(-1), out.view(-1), float(alpha), tile))
    return out
