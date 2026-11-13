"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def gemm_tensor_core_cutile_mma(a, b, bias, out, K: ct.Constant[int],
                               BM: ct.Constant[int], BN: ct.Constant[int], BK: ct.Constant[int]):
    row_tile, col_tile = ct.bid(0), ct.bid(1)
    acc = ct.full((BM, BN), 0.0, dtype=ct.float32)
    for k_tile in range(ct.cdiv(K, BK)):
        left = ct.load(a, (row_tile, k_tile), (BM, BK), padding_mode=ct.PaddingMode.ZERO)
        right = ct.load(b, (k_tile, col_tile), (BK, BN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(left, right, acc)
    cols = col_tile * BN + ct.arange(BN, dtype=ct.int32)
    offset = ct.gather(bias, cols, padding_value=0).astype(ct.float32)
    ct.store(out, (row_tile, col_tile), (acc + ct.reshape(offset, (1, BN))).astype(out.dtype))


def gemm_tensor_core_forward(a, b, bias):
    _cuda_inputs(a, b, bias)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or bias.shape != (b.shape[1],):
        raise ValueError("Tensor Core GEMM requires a [M,K], b [K,N], bias [N]")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16) or bias.dtype != torch.float32:
        raise ValueError("Tensor Core GEMM requires fp16/bf16 operands and fp32 bias")
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    # Fixed shape-only choice; a smaller M tile avoids excess work at M=8.
    bm, bn, bk = (16 if m <= 16 else 64), 128, 64
    if m and n:
        with torch.cuda.device(a.device):
            ct.launch(torch.cuda.current_stream(), ((m + bm - 1) // bm, (n + bn - 1) // bn, 1),
                      gemm_tensor_core_cutile_mma, (a, b, bias, out, k, bm, bn, bk))
    return out
