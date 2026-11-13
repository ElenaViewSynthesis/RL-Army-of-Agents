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
def log_softmax_gather_cutile_rows(logits, token_ids, out,
                                  V: ct.Constant[int], BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32)
    running_max = ct.full((), NEG_INF, dtype=ct.float32)
    normalizer = ct.full((), 0.0, dtype=ct.float32)
    # Online log-sum-exp keeps the live tile at 1024 even for V=128256.
    for tile in range(ct.cdiv(V, BLOCK)):
        cols = tile * BLOCK + offsets
        values = ct.gather(logits, (row, cols), padding_value=NEG_INF).astype(ct.float32)
        next_max = ct.maximum(running_max, ct.max(values, axis=0))
        normalizer = normalizer * ct.exp(running_max - next_max) + ct.sum(ct.exp(values - next_max), axis=0)
        running_max = next_max
    token = ct.gather(token_ids, row)
    selected = ct.gather(logits, (row, token)).astype(ct.float32)
    result = (selected - running_max) - ct.log(normalizer)
    ct.scatter(out, row, result.astype(out.dtype))


def log_softmax_gather_forward(logits, token_ids):
    _cuda_inputs(logits, token_ids)
    if logits.ndim != 2 or logits.shape[1] < 1 or token_ids.shape != (logits.shape[0],):
        raise ValueError("log_softmax_gather requires logits [B,V] and token_ids [B]")
    if logits.dtype not in (torch.float32, torch.bfloat16) or token_ids.dtype != torch.int64:
        raise ValueError("log_softmax_gather requires fp32/bf16 logits and int64 token IDs")
    rows, vocab = logits.shape
    out = torch.empty((rows,), dtype=torch.float32, device=logits.device)
    if rows:
        with torch.cuda.device(logits.device):
            ct.launch(torch.cuda.current_stream(), (rows, 1, 1), log_softmax_gather_cutile_rows,
                      (logits, token_ids, out, vocab, 1024))
    return out
