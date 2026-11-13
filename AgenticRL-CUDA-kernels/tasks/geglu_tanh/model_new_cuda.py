import torch.nn as nn
from kernels.cuda import geglu_tanh_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, x):
        return geglu_tanh_forward(x)
