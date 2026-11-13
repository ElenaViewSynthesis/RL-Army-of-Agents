"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def residual_rmsnorm_cutile_rows(x, residual, weight, out, summed, eps,
                                 H: ct.Constant[int], BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    cols = ct.arange(BLOCK, dtype=ct.int32)
    x_value = ct.gather(x, (row, cols)).astype(ct.float32)
    r_value = ct.gather(residual, (row, cols)).astype(ct.float32)
    # The contract rounds the residual sum BEFORE computing its fp32 RMS.
    rounded = (x_value + r_value).astype(summed.dtype)
    ct.scatter(summed, (row, cols), rounded)
    value = rounded.astype(ct.float32)
    inv = ct.rsqrt(ct.sum(value * value, axis=0) / H + eps)
    scale = ct.gather(weight, cols).astype(ct.float32)
    ct.scatter(out, (row, cols), (value * inv * scale).astype(out.dtype))


def residual_rmsnorm_forward(x, residual, weight, eps=1e-6):
    _cuda_inputs(x, residual, weight)
    if x.ndim != 2 or x.shape != residual.shape or x.shape[1] < 1 or weight.shape != (x.shape[1],):
        raise ValueError("residual RMSNorm requires x/residual [rows,H] and weight [H]")
    if any(t.dtype != torch.bfloat16 for t in (x, residual, weight)):
        raise ValueError("residual RMSNorm requires bf16 inputs")
    rows, h = x.shape
    out, summed = torch.empty_like(x), torch.empty_like(x)
    block = 1 << (h - 1).bit_length()
    if rows:
        with torch.cuda.device(x.device):
            ct.launch(torch.cuda.current_stream(), (rows, 1, 1), residual_rmsnorm_cutile_rows,
                      (x, residual, weight, out, summed, float(eps), h, block))
    return out, summed
