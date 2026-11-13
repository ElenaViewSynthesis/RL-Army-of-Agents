import torch


def silu_and_mul_forward(x):
    import cuda_extension
    if x.ndim != 2:
        raise ValueError("silu_and_mul requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("silu_and_mul requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("silu_and_mul requires a contiguous input")
    return cuda_extension.silu_and_mul_forward(x)
