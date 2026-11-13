import torch.nn as nn
from kernels.triton import geglu_tanh_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, x):
        return geglu_tanh_forward(x)
