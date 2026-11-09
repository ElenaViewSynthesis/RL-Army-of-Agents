import torch
def residual_rmsnorm_forward(x, residual, weight, eps):
    import cuda_extension
    if x.ndim != 2 or residual.shape != x.shape or weight.shape != (x.shape[1],):
        raise ValueError("residual_rmsnorm requires x,residual [B,H] and weight [H]")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("residual_rmsnorm requires bfloat16 inputs")
    if not (x.device == residual.device == weight.device):
        raise ValueError("inputs must share a device")
    if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError("residual_rmsnorm requires contiguous inputs")
    return cuda_extension.residual_rmsnorm_forward(x, residual, weight, float(eps))
