"""Original cuTile implementation; GPU compilation/correctness not validated locally.

Contract adapted from NVIDIA TileGym v1.5.1 (commit bb1a7e72, MIT) (ops/cutile/rope.py, lines 23-121); the kernel is written to this repo's
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
def rotary_embedding_cutile_rows(x, cos, sin, out, S_TILES: ct.Constant[int],
                                 ROWS: ct.Constant[int], HALF: ct.Constant[int]):
    tile = ct.bid(0)
    table = tile % S_TILES
    x1 = ct.load(x, (tile, 0), (ROWS, HALF), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    x2 = ct.load(x, (tile, 1), (ROWS, HALF), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    c = ct.load(cos, (table, 0), (ROWS, HALF), padding_mode=ct.PaddingMode.ZERO)
    s = ct.load(sin, (table, 0), (ROWS, HALF), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (tile, 0), (x1 * c - x2 * s).astype(out.dtype))
    ct.store(out, (tile, 1), (x2 * c + x1 * s).astype(out.dtype))


def _check(q, k, cos, sin):
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("rotary_embedding requires q [B,Hq,S,D] and k [B,Hkv,S,D]")
    s, d = q.shape[2], q.shape[3]
    if d % 2 or cos.shape != (s, d // 2) or sin.shape != (s, d // 2):
        raise ValueError("rotary_embedding requires even D and cos/sin [S,D/2]")
    if q.dtype not in (torch.bfloat16, torch.float32) or k.dtype != q.dtype:
        raise ValueError("rotary_embedding requires float32 or bfloat16 q and k")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("rotary_embedding requires float32 cos/sin tables")
    if not all(t.is_contiguous() for t in (q, k, cos, sin)):
        raise ValueError("rotary_embedding requires contiguous inputs")


def rotary_embedding_forward(q, k, cos, sin):
    _cuda_inputs(q, k, cos, sin)
    _check(q, k, cos, sin)
    seq, half = q.shape[2], q.shape[3] // 2
    # 16-row tiles never straddle a sequence boundary (S is a multiple of 16), so one
    # tile reads one contiguous 16-row slice of the [S, D/2] tables.
    if seq % 16 or half & (half - 1):
        raise ValueError("rotary_embedding requires S % 16 == 0 and D/2 a power of two")
    outputs = []
    for x in (q, k):
        out = torch.empty_like(x)
        rows = x.numel() // (2 * half)
        if rows:
            with torch.cuda.device(x.device):
                ct.launch(torch.cuda.current_stream(), (rows // 16, 1, 1), rotary_embedding_cutile_rows,
                          (x.view(rows, 2 * half), cos, sin, out.view(rows, 2 * half), seq // 16, 16, half))
        outputs.append(out)
    return outputs[0], outputs[1]
