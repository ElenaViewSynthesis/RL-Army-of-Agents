"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def cispo_clip_weight_cutile_tiles(log_ratio, advantage, mask, out, low, high,
                                   S: ct.Constant[int], BLOCK: ct.Constant[int]):
    index = ct.bid(0) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    ratio = ct.exp(ct.gather(log_ratio, index).astype(ct.float32))
    clipped = ct.minimum(ct.maximum(ratio, low), high)
    adv = ct.gather(advantage, index // S).astype(ct.float32)
    valid = ct.gather(mask, index, padding_value=False)
    ct.scatter(out, index, ct.where(valid, clipped * adv, 0.0).astype(out.dtype))


def cispo_clip_weight_forward(log_ratio, advantage, mask, low=0.0, high=5.0):
    _cuda_inputs(log_ratio, advantage, mask)
    if log_ratio.ndim != 2 or log_ratio.shape[1] < 1 or advantage.shape != (log_ratio.shape[0],) or mask.shape != log_ratio.shape:
        raise ValueError("CISPO requires log_ratio/mask [B,S] and advantage [B]")
    if log_ratio.dtype != torch.float32 or advantage.dtype != torch.float32 or mask.dtype != torch.bool:
        raise ValueError("CISPO requires fp32 log_ratio/advantage and a boolean mask")
    out = torch.empty_like(log_ratio)
    count = log_ratio.numel()
    block = 1024
    if count:
        with torch.cuda.device(log_ratio.device):
            ct.launch(torch.cuda.current_stream(), ((count + block - 1) // block, 1, 1),
                      cispo_clip_weight_cutile_tiles,
                      (log_ratio.view(-1), advantage, mask.view(-1), out.view(-1),
                       float(low), float(high), log_ratio.shape[1], block))
    return out
