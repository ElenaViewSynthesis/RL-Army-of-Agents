import torch.nn as nn
from kernels.triton import prefix_sum_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, x):
        return prefix_sum_forward(x)
