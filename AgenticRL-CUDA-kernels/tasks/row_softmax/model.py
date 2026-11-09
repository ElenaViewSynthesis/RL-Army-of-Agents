import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def forward(self, x):
        return torch.softmax(x.float(), dim=-1).to(x.dtype)


def get_inputs():
    return make_inputs("row_softmax")


def get_init_inputs():
    return []
