import torch.nn as nn
from kernels.triton import row_softmax_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, x):
        return row_softmax_forward(x)
