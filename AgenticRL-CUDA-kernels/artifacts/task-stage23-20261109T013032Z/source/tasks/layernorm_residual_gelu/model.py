import math

import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, residual, weight, bias):
        value = x.float()
        centered = value - value.mean(dim=-1, keepdim=True)
        norm = centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + self.eps)
        value = norm * weight.float() + bias.float() + residual.float()
        return (0.5 * value * (1 + torch.erf(value / math.sqrt(2)))).to(x.dtype)


def get_inputs():
    return make_inputs("layernorm_residual_gelu")


def get_init_inputs():
    return [1e-5]
