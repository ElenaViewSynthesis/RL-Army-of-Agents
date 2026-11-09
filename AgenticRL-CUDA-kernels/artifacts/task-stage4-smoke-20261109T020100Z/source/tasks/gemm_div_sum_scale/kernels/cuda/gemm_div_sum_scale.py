import torch


def gemm_div_sum_scale_forward(a, b, divisor, scale):
    import cuda_extension
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm_div_sum_scale requires a [M,K] and b [K,N]")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same dtype and device")
    if a.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("gemm_div_sum_scale requires float32 or bfloat16 inputs")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("gemm_div_sum_scale requires contiguous inputs")
    return cuda_extension.gemm_div_sum_scale_forward(a, b, float(divisor), float(scale))
