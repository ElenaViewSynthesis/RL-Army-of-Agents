import torch
def flash_attention_forward(q, k, v, causal):
    import cuda_extension
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("flash_attention requires q,k,v [B,H,S,D] of equal shape")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("flash_attention requires bfloat16 inputs")
    if q.shape[3] != 64:
        raise ValueError("flash_attention requires head dim D=64")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("flash_attention requires contiguous inputs")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must share a device")
    return cuda_extension.flash_attention_forward(q, k, v, causal)
