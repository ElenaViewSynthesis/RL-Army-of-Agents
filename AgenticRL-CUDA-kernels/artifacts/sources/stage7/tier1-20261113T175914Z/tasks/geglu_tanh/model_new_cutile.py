import torch.nn as nn
from kernels.cutile import geglu_tanh_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def forward(self, x):
        return geglu_tanh_forward(x)
