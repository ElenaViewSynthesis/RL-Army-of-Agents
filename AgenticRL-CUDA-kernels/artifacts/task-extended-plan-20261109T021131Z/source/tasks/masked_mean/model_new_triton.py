import torch.nn as nn
from kernels.triton import masked_mean_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, x, lengths):
        return masked_mean_forward(x, lengths)
