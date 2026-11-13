import torch.nn as nn
from kernels.triton import relu_squared_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, x):
        return relu_squared_forward(x)
