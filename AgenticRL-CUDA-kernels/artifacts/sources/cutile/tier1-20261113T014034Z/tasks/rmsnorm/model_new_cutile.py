import torch.nn as nn
from kernels.cutile import rmsnorm_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        return rmsnorm_forward(x, weight, self.eps)
