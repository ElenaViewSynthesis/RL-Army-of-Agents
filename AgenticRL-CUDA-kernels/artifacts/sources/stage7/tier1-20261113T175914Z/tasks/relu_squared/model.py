import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    """relu(x)**2 in fp32, cast back to the input dtype; a new output tensor."""

    def forward(self, x):
        return x.float().clamp(min=0).square().to(x.dtype)


def get_inputs():
    return make_inputs("relu_squared")


def get_init_inputs():
    return []
