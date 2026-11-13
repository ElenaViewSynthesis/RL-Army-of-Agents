import torch.nn as nn
from kernels.cutile import residual_rmsnorm_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, residual, weight):
        return residual_rmsnorm_forward(x, residual, weight, self.eps)
