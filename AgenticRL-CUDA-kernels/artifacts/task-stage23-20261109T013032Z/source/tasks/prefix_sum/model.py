import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=0, dtype=x.dtype)


def get_inputs():
    return make_inputs("prefix_sum")


def get_init_inputs():
    return []
