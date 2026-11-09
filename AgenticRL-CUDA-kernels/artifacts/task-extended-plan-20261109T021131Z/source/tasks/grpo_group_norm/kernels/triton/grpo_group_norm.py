import os

import torch
import triton
import triton.language as tl


@triton.jit
def grpo_group_norm_kernel(x_ptr, out_ptr, g, eps, BLOCK_G: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_G)
    mask = offs < g
    x = tl.load(x_ptr + row * g + offs, mask=mask, other=0.0)
    mean = tl.sum(x) / g
    centered = tl.where(mask, x - mean, 0.0)
    std = tl.sqrt(tl.sum(centered * centered) / g)
    tl.store(out_ptr + row * g + offs, centered / (std + eps), mask=mask)


def grpo_group_norm_forward(rewards, eps):
    if rewards.ndim != 2 or rewards.dtype != torch.float32:
        raise ValueError("grpo_group_norm requires float32 [B,G] rewards")
    if not rewards.is_contiguous():
        raise ValueError("grpo_group_norm requires contiguous input")
    if rewards.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, g = rewards.shape
    out = torch.empty_like(rewards)
    if b and g:
        grpo_group_norm_kernel[(b,)](rewards, out, g, float(eps),
                                     BLOCK_G=triton.next_power_of_2(g))
    return out
