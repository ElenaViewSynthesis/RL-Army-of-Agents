import torch.nn as nn
from kernels.cutile import relu_squared_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def forward(self, x):
        return relu_squared_forward(x)
