import torch.nn as nn
from kernels.cuda import rmsnorm_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        return rmsnorm_forward(x, weight, self.eps)
