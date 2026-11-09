import torch


def fused_adamw_forward(parameter, gradient, first_moment, second_moment,
                        lr, beta1, beta2, eps, weight_decay, step):
    import cuda_extension
    inputs = (parameter, gradient, first_moment, second_moment)
    if any(x.dtype != torch.float32 for x in inputs) or len({x.shape for x in inputs}) != 1:
        raise ValueError("fused_adamw requires four same-shape float32 tensors")
    if len({x.device for x in inputs}) != 1:
        raise ValueError("fused_adamw inputs must share a device")
    if any(not x.is_contiguous() for x in inputs):
        raise ValueError("fused_adamw requires contiguous inputs")
    return cuda_extension.fused_adamw_forward(parameter, gradient, first_moment,
                                              second_moment, float(lr), float(beta1),
                                              float(beta2), float(eps),
                                              float(weight_decay), float(step))
