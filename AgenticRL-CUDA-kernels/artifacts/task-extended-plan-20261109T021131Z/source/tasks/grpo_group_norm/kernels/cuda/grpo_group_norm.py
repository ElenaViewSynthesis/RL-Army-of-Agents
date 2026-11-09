import torch


def grpo_group_norm_forward(rewards, eps):
    import cuda_extension
    if rewards.ndim != 2 or rewards.dtype != torch.float32:
        raise ValueError("grpo_group_norm requires float32 [B,G] rewards")
    if not rewards.is_contiguous():
        raise ValueError("grpo_group_norm requires contiguous input")
    return cuda_extension.grpo_group_norm_forward(rewards, float(eps))
