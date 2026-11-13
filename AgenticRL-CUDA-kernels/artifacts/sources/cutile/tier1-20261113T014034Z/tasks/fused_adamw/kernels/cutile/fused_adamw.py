"""Original cuTile implementation; GPU compilation/correctness not validated locally."""

import cuda.tile as ct
import torch


def _cuda_inputs(*tensors):
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("cuTile requires CUDA tensors; no CPU interpreter")
    if any(t.device != tensors[0].device or not t.is_contiguous() for t in tensors):
        raise ValueError("cuTile inputs must be contiguous and share a device")


@ct.kernel
def fused_adamw_cutile_tiles(parameter, gradient, first, second, updated, new_m, new_v,
                             lr, beta1, beta2, eps, decay, correction1, correction2,
                             BLOCK: ct.Constant[int]):
    index = ct.bid(0) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    p = ct.gather(parameter, index).astype(ct.float32)
    g = ct.gather(gradient, index).astype(ct.float32)
    old_m = ct.gather(first, index).astype(ct.float32)
    old_v = ct.gather(second, index).astype(ct.float32)
    m = beta1 * old_m + (1.0 - beta1) * g
    v = beta2 * old_v + (1.0 - beta2) * g * g
    value = p * (1.0 - lr * decay) - lr * (m / correction1) / (ct.sqrt(v / correction2) + eps)
    ct.scatter(updated, index, value.astype(updated.dtype))
    ct.scatter(new_m, index, m.astype(new_m.dtype))
    ct.scatter(new_v, index, v.astype(new_v.dtype))


def fused_adamw_forward(parameter, gradient, first_moment, second_moment,
                        lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=10):
    _cuda_inputs(parameter, gradient, first_moment, second_moment)
    if any(t.shape != parameter.shape or t.dtype != torch.float32 for t in (parameter, gradient, first_moment, second_moment)):
        raise ValueError("AdamW requires four matching fp32 arrays")
    updated, new_m, new_v = (torch.empty_like(parameter), torch.empty_like(first_moment),
                            torch.empty_like(second_moment))
    count = parameter.numel()
    block = 1024
    if count:
        with torch.cuda.device(parameter.device):
            ct.launch(torch.cuda.current_stream(), ((count + block - 1) // block, 1, 1),
                      fused_adamw_cutile_tiles,
                      (parameter.view(-1), gradient.view(-1), first_moment.view(-1), second_moment.view(-1),
                       updated.view(-1), new_m.view(-1), new_v.view(-1), float(lr), float(beta1),
                       float(beta2), float(eps), float(weight_decay), 1.0 - beta1 ** step,
                       1.0 - beta2 ** step, block))
    return updated, new_m, new_v
