"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


NEG_INF = -float("inf")


@ct.kernel
def row_softmax_cutile_rows(x, out, BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    cols = ct.arange(BLOCK, dtype=ct.int32)
    value = ct.gather(x, (row, cols), padding_value=NEG_INF).astype(ct.float32)
    numer = ct.exp(value - ct.max(value, axis=0))
    result = numer / ct.sum(numer, axis=0)
    ct.scatter(out, (row, cols), result.astype(out.dtype))


def row_softmax_forward(x):
    _cuda_inputs(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("row_softmax requires x [rows,H], H > 0")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("row_softmax requires float16/bfloat16/float32")
    rows, h = x.shape
    block = 1 << (h - 1).bit_length()
    out = torch.empty_like(x)
    if rows:
        with torch.cuda.device(x.device):
            ct.launch(torch.cuda.current_stream(), (rows, 1, 1), row_softmax_cutile_rows,
                      (x, out, block))
    return out
