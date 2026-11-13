import torch


def relu_squared_forward(x):
    import cuda_extension
    if x.ndim != 2:
        raise ValueError("relu_squared requires a 2-D input")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("relu_squared requires float32 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("relu_squared requires a contiguous input")
    return cuda_extension.relu_squared_forward(x)
