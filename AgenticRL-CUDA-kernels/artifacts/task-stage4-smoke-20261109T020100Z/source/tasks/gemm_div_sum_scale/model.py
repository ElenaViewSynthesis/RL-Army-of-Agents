import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, divisor=2.0, scale=1.5):
        super().__init__()
        self.divisor, self.scale = divisor, scale
        torch.set_float32_matmul_precision("highest")

    def forward(self, a, b):
        product = torch.matmul(a.float(), b.float())
        return (product / self.divisor).sum(dim=1, keepdim=True) * self.scale


def get_inputs():
    return make_inputs("gemm_div_sum_scale")


def get_init_inputs():
    return [2.0, 1.5]
