import torch


def layernorm_residual_gelu_forward(x, residual, weight, bias, eps):
    import cuda_extension
    if x.ndim != 2:
        raise ValueError("layernorm_residual_gelu requires x [B,H]")
    for name, t in (("residual", residual), ("weight", weight), ("bias", bias)):
        if t.device != x.device:
            raise ValueError(f"{name} must share x's device")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if residual.shape != x.shape or weight.shape != (x.shape[1],) or bias.shape != (x.shape[1],):
        raise ValueError("shape mismatch against x [B,H], weight/bias [H]")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("layernorm_residual_gelu requires float32 or bfloat16 input")
    return cuda_extension.layernorm_residual_gelu_forward(x, residual, weight, bias,
                                                          float(eps))
