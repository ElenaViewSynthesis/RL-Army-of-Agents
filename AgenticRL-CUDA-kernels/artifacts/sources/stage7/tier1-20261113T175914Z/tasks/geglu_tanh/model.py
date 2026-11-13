import math

import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    """x [B,2H] -> x[:, :H] * gelu_tanh(x[:, H:]); tanh-approximate GELU on the SECOND half."""

    def forward(self, x):
        h = x.shape[-1] // 2
        a, b = x[:, :h].float(), x[:, h:].float()
        gelu = 0.5 * b * (1 + torch.tanh(math.sqrt(2 / math.pi) * (b + 0.044715 * b ** 3)))
        return (a * gelu).to(x.dtype)


def get_inputs():
    return make_inputs("geglu_tanh")


def get_init_inputs():
    return []
