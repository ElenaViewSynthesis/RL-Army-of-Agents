import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_float32_matmul_precision("highest")

    def forward(self, a, b, bias):
        return torch.matmul(a.float(), b.float()) + bias.float()


def get_inputs():
    return make_inputs("gemm_tensor_core")


def get_init_inputs():
    return []
