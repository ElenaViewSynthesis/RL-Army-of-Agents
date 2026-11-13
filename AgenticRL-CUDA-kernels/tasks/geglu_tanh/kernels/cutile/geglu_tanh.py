"""Original cuTile implementation; GPU compilation/correctness not validated locally.

Contract adapted from NVIDIA TileGym v1.5.1 (commit bb1a7e72, MIT) (ops/cutile/activation/geglu.py, lines 69-115, and gelu.py, lines 74-82); the kernel is written to this repo's
rules: fp32 arithmetic, fresh outputs, no autotuning, no capability branches or hints.
"""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def geglu_tanh_cutile_chunks(x, out, H: ct.Constant[int], BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    cols = ct.bid(1) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    a = ct.gather(x, (row, cols), padding_value=0).astype(ct.float32)
    b = ct.gather(x, (row, cols + H), padding_value=0).astype(ct.float32)
    inner = 0.7978845608028654 * (b + 0.044715 * b * b * b)
    # tanh(z) = 1 - 2 / (exp(2z) + 1): exp is already H200-validated here; ct.tanh is not.
    tanh = 1.0 - 2.0 / (ct.exp(2.0 * inner) + 1.0)
    ct.scatter(out, (row, cols), (a * 0.5 * b * (1.0 + tanh)).astype(out.dtype))


def geglu_tanh_forward(x):
    _cuda_inputs(x)
    if x.ndim != 2:
        raise ValueError("geglu_tanh requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("geglu_tanh requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("geglu_tanh requires a contiguous input")
    if x.shape[1] % 2:
        raise ValueError("geglu_tanh requires an even last dimension [B,2H]")
    rows, h = x.shape[0], x.shape[1] // 2
    out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    if rows and h:
        with torch.cuda.device(x.device):
            ct.launch(torch.cuda.current_stream(), (rows, (h + 1023) // 1024, 1),
                      geglu_tanh_cutile_chunks, (x, out, h, 1024))
    return out
