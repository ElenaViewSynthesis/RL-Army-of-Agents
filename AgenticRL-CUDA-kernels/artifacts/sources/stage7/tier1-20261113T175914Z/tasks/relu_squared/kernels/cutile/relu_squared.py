"""Original cuTile implementation; GPU compilation/correctness not validated locally.

Contract adapted from NVIDIA TileGym v1.5.1 (commit bb1a7e72, MIT) (suites/liger/cutile/relu_squared.py, lines 55-72); the kernel is written to this repo's
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
def relu_squared_cutile_chunks(x, out, BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    cols = ct.bid(1) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    value = ct.gather(x, (row, cols), padding_value=0).astype(ct.float32)
    positive = ct.maximum(value, ct.full((), 0.0, dtype=ct.float32))
    ct.scatter(out, (row, cols), (positive * positive).astype(out.dtype))


def relu_squared_forward(x):
    _cuda_inputs(x)
    if x.ndim != 2:
        raise ValueError("relu_squared requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("relu_squared requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("relu_squared requires a contiguous input")
    rows, h = x.shape
    out = torch.empty_like(x)
    if rows and h:
        with torch.cuda.device(x.device):
            ct.launch(torch.cuda.current_stream(), (rows, (h + 1023) // 1024, 1),
                      relu_squared_cutile_chunks, (x, out, 1024))
    return out
