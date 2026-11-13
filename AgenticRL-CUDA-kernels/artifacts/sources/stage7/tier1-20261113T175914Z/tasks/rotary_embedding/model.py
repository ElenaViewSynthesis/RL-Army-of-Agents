import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    c, s = cos[None, None], sin[None, None]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)


class Model(nn.Module):
    """Half-split (Llama) RoPE; returns NEW (q_rot, k_rot); q, k, cos and sin are read-only."""

    def forward(self, q, k, cos, sin):
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def get_inputs():
    return make_inputs("rotary_embedding")


def get_init_inputs():
    return []
