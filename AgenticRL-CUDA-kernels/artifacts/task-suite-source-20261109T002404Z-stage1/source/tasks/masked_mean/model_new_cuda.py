import torch.nn as nn
from kernels.cuda import masked_mean_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, x, lengths):
        return masked_mean_forward(x, lengths)
