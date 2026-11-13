import torch


def geglu_tanh_forward(x):
    import cuda_extension
    if x.ndim != 2:
        raise ValueError("geglu_tanh requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("geglu_tanh requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("geglu_tanh requires a contiguous input")
    return cuda_extension.geglu_tanh_forward(x)
