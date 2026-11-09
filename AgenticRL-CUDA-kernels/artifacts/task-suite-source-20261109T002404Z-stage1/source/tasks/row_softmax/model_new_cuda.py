import torch.nn as nn
from kernels.cuda import row_softmax_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, x):
        return row_softmax_forward(x)
