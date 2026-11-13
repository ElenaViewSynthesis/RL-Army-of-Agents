import torch.nn as nn
from kernels.cutile import row_softmax_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def forward(self, x):
        return row_softmax_forward(x)
