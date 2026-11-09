import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_float32_matmul_precision("highest")

    def forward(self, matrices_a, matrices_b):
        return [torch.matmul(a.float(), b.float()) for a, b in zip(matrices_a, matrices_b)]


def get_inputs():
    return make_inputs("grouped_gemm")


def get_init_inputs():
    return []
