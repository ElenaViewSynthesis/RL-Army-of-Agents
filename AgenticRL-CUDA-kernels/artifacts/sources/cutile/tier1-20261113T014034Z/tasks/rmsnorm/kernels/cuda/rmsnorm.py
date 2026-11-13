import torch


def rmsnorm_forward(x, weight, eps):
    import cuda_extension
    if x.ndim != 2 or weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("rmsnorm requires x [B,H] and weight [H]")
    if x.device != weight.device:
        raise ValueError("x and weight must have the same device")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("rmsnorm requires float32 or bfloat16 input")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("rmsnorm requires contiguous inputs")
    return cuda_extension.rmsnorm_forward(x, weight, float(eps))
