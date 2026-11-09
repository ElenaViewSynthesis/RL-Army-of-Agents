import torch
def decode_attention_forward(q, k, v):
    import cuda_extension
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("decode_attention requires q [B,Hq,1,D] and k,v [B,Hkv,S,D]")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3] or q.shape[2] != 1:
        raise ValueError("decode_attention shape mismatch")
    if k.shape[1] <= 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("query heads must divide evenly over kv heads")
    if q.shape[3] != 64:
        raise ValueError("decode_attention requires head dim D=64")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("decode_attention requires bfloat16 inputs")
    if not (q.device == k.device == v.device):
        raise ValueError("inputs must share a device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("decode_attention requires contiguous inputs")
    return cuda_extension.decode_attention_forward(q, k, v)
