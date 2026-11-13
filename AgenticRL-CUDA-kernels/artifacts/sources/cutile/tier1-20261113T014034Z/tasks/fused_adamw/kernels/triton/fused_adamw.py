import os

import torch
import triton
import triton.language as tl


@triton.jit
def fused_adamw_kernel(p_ptr, g_ptr, m_ptr, v_ptr, updated_ptr, new_m_ptr, new_v_ptr,
                       lr, beta1, beta2, eps, weight_decay, bc1, bc2, n,
                       BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    m_hat = m / bc1
    v_hat = v / bc2
    updated = p * (1.0 - lr * weight_decay) - lr * m_hat / (tl.sqrt(v_hat) + eps)
    tl.store(updated_ptr + offs, updated, mask=mask)
    tl.store(new_m_ptr + offs, m, mask=mask)
    tl.store(new_v_ptr + offs, v, mask=mask)


def fused_adamw_forward(parameter, gradient, first_moment, second_moment,
                        lr, beta1, beta2, eps, weight_decay, step):
    inputs = (parameter, gradient, first_moment, second_moment)
    if any(x.dtype != torch.float32 for x in inputs) or len({x.shape for x in inputs}) != 1:
        raise ValueError("fused_adamw requires four same-shape float32 tensors")
    if len({x.device for x in inputs}) != 1:
        raise ValueError("fused_adamw inputs must share a device")
    if any(not x.is_contiguous() for x in inputs):
        raise ValueError("fused_adamw requires contiguous inputs")
    if parameter.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("CPU execution requires TRITON_INTERPRET=1")
    n = parameter.numel()
    updated = torch.empty_like(parameter)
    new_m = torch.empty_like(first_moment)
    new_v = torch.empty_like(second_moment)
    if n:
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        fused_adamw_kernel[(triton.cdiv(n, 1024),)](
            parameter, gradient, first_moment, second_moment, updated, new_m, new_v,
            float(lr), float(beta1), float(beta2), float(eps), float(weight_decay),
            float(bc1), float(bc2), n, BLOCK=1024)
    return updated, new_m, new_v
