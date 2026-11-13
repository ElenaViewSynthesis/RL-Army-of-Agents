import torch.nn as nn
from kernels.cuda import relu_squared_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, x):
        return relu_squared_forward(x)
