import os

import torch
import triton
import triton.language as tl


@triton.jit
def cispo_clip_weight_kernel(lr_ptr, adv_ptr, mask_ptr, out_ptr, low, high, n, s,
                             BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    lr = tl.load(lr_ptr + offs, mask=mask, other=0.0)
    adv = tl.load(adv_ptr + offs // s, mask=mask, other=0.0)
    m = tl.load(mask_ptr + offs, mask=mask, other=0)
    w = tl.clamp(tl.exp(lr), low, high) * adv
    tl.store(out_ptr + offs, tl.where(m != 0, w, 0.0), mask=mask)


def cispo_clip_weight_forward(log_ratio, advantage, mask, low, high):
    if log_ratio.ndim != 2 or advantage.ndim != 1 or mask.shape != log_ratio.shape:
        raise ValueError("cispo_clip_weight requires log_ratio [B,S], advantage [B], mask [B,S]")
    if advantage.shape[0] != log_ratio.shape[0]:
        raise ValueError("advantage must have one entry per row")
    if log_ratio.dtype != torch.float32 or advantage.dtype != torch.float32:
        raise ValueError("cispo_clip_weight requires float32 log_ratio and advantage")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if not (log_ratio.device == advantage.device == mask.device):
        raise ValueError("inputs must share a device")
    if not log_ratio.is_contiguous() or not advantage.is_contiguous() or not mask.is_contiguous():
        raise ValueError("cispo_clip_weight requires contiguous inputs")
    if log_ratio.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    b, s = log_ratio.shape
    n = b * s
    out = torch.empty_like(log_ratio)
    if n:
        cispo_clip_weight_kernel[(triton.cdiv(n, 1024),)](log_ratio, advantage, mask, out,
                                                          float(low), float(high), n, s,
                                                          BLOCK=1024)
    return out
