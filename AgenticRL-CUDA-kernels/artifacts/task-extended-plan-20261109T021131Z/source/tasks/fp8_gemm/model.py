import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def forward(self, a, b, scale_a, scale_b):
        return torch.matmul(a.float(), b.float()) * scale_a[:, None] * scale_b[None, :]


def get_inputs():
    return make_inputs("fp8_gemm")


def get_init_inputs():
    return []
