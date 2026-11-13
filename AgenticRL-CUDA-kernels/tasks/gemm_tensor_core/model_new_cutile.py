import torch.nn as nn
from kernels.cutile import gemm_tensor_core_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, bias):
        return gemm_tensor_core_forward(a, b, bias)
