"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def rmsnorm_cutile_rows(x, weight, out, eps, H: ct.Constant[int], BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    cols = ct.arange(BLOCK, dtype=ct.int32)
    value = ct.gather(x, (row, cols), padding_value=0).astype(ct.float32)
    scale = ct.gather(weight, cols, padding_value=0).astype(ct.float32)
    inv = ct.rsqrt(ct.sum(value * value, axis=0) / H + eps)
    ct.scatter(out, (row, cols), (value * inv * scale).astype(out.dtype))


def rmsnorm_forward(x, weight, eps=1e-6):
    _cuda_inputs(x, weight)
    if x.ndim != 2 or weight.shape != (x.shape[1],) or x.shape[1] < 1:
        raise ValueError("rmsnorm requires x [rows,H], weight [H], H > 0")
    if x.dtype not in (torch.float32, torch.bfloat16) or weight.dtype != x.dtype:
        raise ValueError("rmsnorm requires matching float32/bfloat16 inputs")
    rows, h = x.shape
    # One complete row tile (at most 8192 for this catalog), no scratch/pass.
    block = 1 << (h - 1).bit_length()
    out = torch.empty_like(x)
    if rows:
        with torch.cuda.device(x.device):
            ct.launch(torch.cuda.current_stream(), (rows, 1, 1), rmsnorm_cutile_rows,
                      (x, weight, out, float(eps), h, block))
    return out
