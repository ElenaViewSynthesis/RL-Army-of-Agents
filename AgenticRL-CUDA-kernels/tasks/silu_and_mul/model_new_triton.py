import torch.nn as nn
from kernels.triton import silu_and_mul_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, x):
        return silu_and_mul_forward(x)
