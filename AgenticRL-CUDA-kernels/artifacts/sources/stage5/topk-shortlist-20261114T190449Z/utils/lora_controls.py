"""Deliberately incorrect multi-LoRA controls; never used by honest solutions."""


def control(name, variant, *args):
    import torch
    from utils.lora_reference import forward
    if variant == "fp16_intermediate":
        return forward(name, *args, half=True)
    out = forward(name, *args)
    if variant == "wrong_padding":
        # Accidentally reduce padded weights into a live output: NaNs must propagate.
        weights = args[1] if "fused" not in name else args[2]
        poison = weights.float().sum()
        return out + poison
    idx = args[4] if "fused" in name else args[2] if "shrink" in name else args[3]
    if "shrink" in name:
        return out + (idx < 0)[None, :, None]*100000
    return out + (idx < 0)[:, None]*100000
