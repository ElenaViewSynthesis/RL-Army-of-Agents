"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def grpo_group_norm_cutile_rows(rewards, out, eps, G: ct.Constant[int], BLOCK: ct.Constant[int]):
    row = ct.bid(0)
    cols = ct.arange(BLOCK, dtype=ct.int32)
    values = ct.gather(rewards, (row, cols), padding_value=0).astype(ct.float32)
    mean = ct.sum(values, axis=0) / G
    centered = ct.where(cols < G, values - mean, 0.0)
    variance = ct.sum(centered * centered, axis=0) / G
    normalized = centered / (ct.sqrt(variance) + eps)
    ct.scatter(out, (row, cols), normalized.astype(out.dtype))


def grpo_group_norm_forward(rewards, eps=1e-6):
    _cuda_inputs(rewards)
    if rewards.ndim != 2 or rewards.shape[1] < 1 or rewards.dtype != torch.float32:
        raise ValueError("GRPO requires fp32 rewards [B,G], G > 0")
    rows, group = rewards.shape
    block = 1 << (group - 1).bit_length()
    out = torch.empty_like(rewards)
    if rows:
        with torch.cuda.device(rewards.device):
            ct.launch(torch.cuda.current_stream(), (rows, 1, 1), grpo_group_norm_cutile_rows,
                      (rewards, out, float(eps), group, block))
    return out
