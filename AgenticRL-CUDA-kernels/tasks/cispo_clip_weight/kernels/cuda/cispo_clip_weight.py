import torch


def cispo_clip_weight_forward(log_ratio, advantage, mask, low, high):
    import cuda_extension
    if log_ratio.ndim != 2 or advantage.ndim != 1 or mask.shape != log_ratio.shape:
        raise ValueError("cispo_clip_weight requires log_ratio [B,S], advantage [B], mask [B,S]")
    if advantage.shape[0] != log_ratio.shape[0]:
        raise ValueError("advantage must have one entry per row")
    if log_ratio.dtype != torch.float32 or advantage.dtype != torch.float32:
        raise ValueError("cispo_clip_weight requires float32 log_ratio and advantage")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if not (log_ratio.device == advantage.device == mask.device):
        raise ValueError("inputs must share a device")
    if not log_ratio.is_contiguous() or not advantage.is_contiguous() or not mask.is_contiguous():
        raise ValueError("cispo_clip_weight requires contiguous inputs")
    return cuda_extension.cispo_clip_weight_forward(log_ratio, advantage, mask,
                                                     float(low), float(high))
