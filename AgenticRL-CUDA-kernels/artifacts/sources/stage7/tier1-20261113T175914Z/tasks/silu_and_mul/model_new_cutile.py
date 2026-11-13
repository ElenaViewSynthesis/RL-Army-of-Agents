import torch.nn as nn
from kernels.cutile import silu_and_mul_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def forward(self, x):
        return silu_and_mul_forward(x)
