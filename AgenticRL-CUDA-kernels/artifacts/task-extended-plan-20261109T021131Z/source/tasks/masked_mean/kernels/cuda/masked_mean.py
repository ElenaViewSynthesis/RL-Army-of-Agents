import torch


def masked_mean_forward(x, lengths):
    import cuda_extension
    if x.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
        raise ValueError("masked_mean requires x [B,L] and lengths [B]")
    if x.device != lengths.device:
        raise ValueError("x and lengths must have the same device")
    if lengths.dtype != torch.int32:
        raise ValueError("masked_mean requires int32 lengths")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("masked_mean requires float32 or bfloat16 input")
    if not x.is_contiguous() or not lengths.is_contiguous():
        raise ValueError("masked_mean requires contiguous inputs")
    return cuda_extension.masked_mean_forward(x, lengths)
