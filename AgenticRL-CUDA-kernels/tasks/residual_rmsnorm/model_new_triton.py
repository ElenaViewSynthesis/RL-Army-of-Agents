import torch.nn as nn
from kernels.triton import residual_rmsnorm_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, residual, weight):
        return residual_rmsnorm_forward(x, residual, weight, self.eps)
