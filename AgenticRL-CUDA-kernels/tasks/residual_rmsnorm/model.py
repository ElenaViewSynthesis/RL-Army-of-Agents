import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, residual, weight):
        summed = (x.float() + residual.float()).to(x.dtype)
        value = summed.float()
        norm = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return (norm * weight.float()).to(x.dtype), summed


def get_inputs():
    return make_inputs("residual_rmsnorm")


def get_init_inputs():
    return [1e-6]
