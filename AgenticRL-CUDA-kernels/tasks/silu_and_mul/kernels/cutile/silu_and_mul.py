"""Original cuTile implementation; GPU compilation/correctness not validated locally.

Contract adapted from NVIDIA TileGym v1.5.1 (commit bb1a7e72, MIT) (ops/cutile/silu_and_mul.py, lines 29-61); the kernel is written to this repo's
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
def silu_and_mul_cutile_chunks(x, out, H: ct.Constant[int], BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    cols = ct.bid(1) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    gate = ct.gather(x, (row, cols), padding_value=0).astype(ct.float32)
    up = ct.gather(x, (row, cols + H), padding_value=0).astype(ct.float32)
    silu = gate / (1.0 + ct.exp(-gate))
    ct.scatter(out, (row, cols), (silu * up).astype(out.dtype))


def silu_and_mul_forward(x):
    _cuda_inputs(x)
    if x.ndim != 2:
        raise ValueError("silu_and_mul requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("silu_and_mul requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("silu_and_mul requires a contiguous input")
    if x.shape[1] % 2:
        raise ValueError("silu_and_mul requires an even last dimension [B,2H]")
    rows, h = x.shape[0], x.shape[1] // 2
    out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    if rows and h:
        with torch.cuda.device(x.device):
            ct.launch(torch.cuda.current_stream(), (rows, (h + 1023) // 1024, 1),
                      silu_and_mul_cutile_chunks, (x, out, h, 1024))
    return out
