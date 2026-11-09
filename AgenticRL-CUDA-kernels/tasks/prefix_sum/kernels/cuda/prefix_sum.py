import torch


def prefix_sum_forward(x):
    import cuda_extension
    if x.ndim != 1:
        raise ValueError("prefix_sum requires a 1D input")
    if x.dtype not in (torch.int32, torch.float32):
        raise ValueError("prefix_sum requires int32 or float32 input")
    if not x.is_contiguous():
        raise ValueError("prefix_sum requires contiguous input")
    return cuda_extension.prefix_sum_forward(x)
