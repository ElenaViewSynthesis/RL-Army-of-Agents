import torch
def topk_sampling_forward(logits, uniforms, top_k, top_p):
    import cuda_extension
    if logits.ndim != 2 or uniforms.ndim != 1 or uniforms.shape[0] != logits.shape[0]:
        raise ValueError("topk_sampling requires logits [B,V] and uniforms [B]")
    if logits.dtype != torch.float32 or uniforms.dtype != torch.float32:
        raise ValueError("topk_sampling requires float32 inputs")
    if logits.device != uniforms.device:
        raise ValueError("inputs must share a device")
    if not logits.is_contiguous() or not uniforms.is_contiguous():
        raise ValueError("topk_sampling requires contiguous inputs")
    if logits.shape[1] <= 0 or not (1 <= top_k <= min(32, logits.shape[1])):
        raise ValueError("top_k must be in [1,min(32,V)] and V>0")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0,1]")
    return cuda_extension.topk_sampling_forward(logits, uniforms, int(top_k), float(top_p))
