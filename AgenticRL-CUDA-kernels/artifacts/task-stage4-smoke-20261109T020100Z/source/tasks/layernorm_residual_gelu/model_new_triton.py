import torch.nn as nn
from kernels.triton import layernorm_residual_gelu_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, residual, weight, bias):
        return layernorm_residual_gelu_forward(x, residual, weight, bias, self.eps)
