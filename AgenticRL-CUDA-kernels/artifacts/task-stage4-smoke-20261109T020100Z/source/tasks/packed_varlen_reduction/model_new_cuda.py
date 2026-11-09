import torch.nn as nn
from kernels.cuda import packed_varlen_reduction_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, values, offsets):
        return packed_varlen_reduction_forward(values, offsets, self.reduction)
