import torch.nn as nn
from kernels.cuda import silu_and_mul_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, x):
        return silu_and_mul_forward(x)
