import torch.nn as nn
from kernels.cuda import prefix_sum_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, x):
        return prefix_sum_forward(x)
