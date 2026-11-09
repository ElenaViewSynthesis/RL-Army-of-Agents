import torch


def log_softmax_gather_forward(logits, token_ids):
    import cuda_extension
    if logits.ndim != 2 or token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError("log_softmax_gather requires logits [B,V] and token_ids [B]")
    if token_ids.dtype != torch.int64:
        raise ValueError("log_softmax_gather requires int64 token_ids")
    if logits.device != token_ids.device:
        raise ValueError("logits and token_ids must have the same device")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("log_softmax_gather requires float32 or bfloat16 input")
    if not logits.is_contiguous() or not token_ids.is_contiguous():
        raise ValueError("log_softmax_gather requires contiguous inputs")
    return cuda_extension.log_softmax_gather_forward(logits, token_ids)
