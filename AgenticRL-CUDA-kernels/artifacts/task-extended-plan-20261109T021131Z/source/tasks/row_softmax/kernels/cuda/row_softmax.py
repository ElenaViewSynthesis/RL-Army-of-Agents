import torch


def row_softmax_forward(x):
    import cuda_extension
    if x.ndim != 2:
        raise ValueError("row_softmax requires a 2D input")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("row_softmax requires float16, bfloat16 or float32 input")
    if not x.is_contiguous():
        raise ValueError("row_softmax requires contiguous input")
    return cuda_extension.row_softmax_forward(x)
