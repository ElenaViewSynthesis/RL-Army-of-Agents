import torch


def _check(q, k, cos, sin):
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("rotary_embedding requires q [B,Hq,S,D] and k [B,Hkv,S,D]")
    s, d = q.shape[2], q.shape[3]
    if d % 2 or cos.shape != (s, d // 2) or sin.shape != (s, d // 2):
        raise ValueError("rotary_embedding requires even D and cos/sin [S,D/2]")
    if q.dtype not in (torch.bfloat16, torch.float32) or k.dtype != q.dtype:
        raise ValueError("rotary_embedding requires float32 or bfloat16 q and k")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("rotary_embedding requires float32 cos/sin tables")
    if not all(t.is_contiguous() for t in (q, k, cos, sin)):
        raise ValueError("rotary_embedding requires contiguous inputs")


def rotary_embedding_forward(q, k, cos, sin):
    import cuda_extension
    _check(q, k, cos, sin)
    return cuda_extension.rotary_embedding_forward(q, k, cos, sin)
