"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def gemm_div_sum_scale_cutile_mma(a, b, product, K: ct.Constant[int],
                                 BM: ct.Constant[int], BN: ct.Constant[int], BK: ct.Constant[int]):
    row_tile, col_tile = ct.bid(0), ct.bid(1)
    acc = ct.full((BM, BN), 0.0, dtype=ct.float32)
    for k_tile in range(ct.cdiv(K, BK)):
        left = ct.load(a, (row_tile, k_tile), (BM, BK), padding_mode=ct.PaddingMode.ZERO)
        right = ct.load(b, (k_tile, col_tile), (BK, BN), padding_mode=ct.PaddingMode.ZERO)
        # Native fp32 stays fp32 (not tf32); bf16 products accumulate in fp32.
        acc = ct.mma(left, right, acc)
    ct.store(product, (row_tile, col_tile), acc)


@ct.kernel
def gemm_div_sum_scale_cutile_reduce(product, out, divisor, scale,
                                    N: ct.Constant[int], BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    total = ct.full((), 0.0, dtype=ct.float32)
    offsets = ct.arange(BLOCK, dtype=ct.int32)
    for n_tile in range(ct.cdiv(N, BLOCK)):
        cols = n_tile * BLOCK + offsets
        values = ct.gather(product, (row, cols), padding_value=0).astype(ct.float32)
        total = total + ct.sum(values / divisor, axis=0)
    ct.scatter(out, row, (total * scale).astype(out.dtype))


def gemm_div_sum_scale_forward(a, b, divisor=2.0, scale=1.5):
    _cuda_inputs(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("GEMM requires a [M,K] and b [K,N]")
    if a.dtype != b.dtype or a.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("GEMM divide/sum requires matching fp32/bf16 inputs")
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    product = torch.empty((m, n), dtype=torch.float32, device=a.device)
    out = torch.empty((m, 1), dtype=torch.float32, device=a.device)
    bm, bn, bk = 32, 64, 32
    with torch.cuda.device(a.device):
        if m and n:
            ct.launch(torch.cuda.current_stream(), ((m + bm - 1) // bm, (n + bn - 1) // bn, 1),
                      gemm_div_sum_scale_cutile_mma, (a, b, product, k, bm, bn, bk))
        if m:
            ct.launch(torch.cuda.current_stream(), (m, 1, 1), gemm_div_sum_scale_cutile_reduce,
                      (product, out.view(-1), float(divisor), float(scale), n, 1024))
    return out
