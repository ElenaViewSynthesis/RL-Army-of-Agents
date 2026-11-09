import torch


def gemm_tensor_core_forward(a, b, bias):
    import cuda_extension
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm_tensor_core requires a [M,K] and b [K,N]")
    if a.dtype != b.dtype or a.device != b.device or a.device != bias.device:
        raise ValueError("a, b and bias must share dtype and device")
    if bias.ndim != 1 or bias.shape[0] != b.shape[1] or bias.dtype != torch.float32:
        raise ValueError("bias must be float32 [N]")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("gemm_tensor_core requires float16 or bfloat16 inputs")
    if not a.is_contiguous() or not b.is_contiguous() or not bias.is_contiguous():
        raise ValueError("gemm_tensor_core requires contiguous inputs")
    return cuda_extension.gemm_tensor_core_forward(a, b, bias)
