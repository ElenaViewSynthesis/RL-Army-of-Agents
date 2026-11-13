import torch.nn as nn
from kernels.cutile import cispo_clip_weight_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def __init__(self, low=0.0, high=5.0):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, log_ratio, advantage, mask):
        return cispo_clip_weight_forward(log_ratio, advantage, mask, self.low, self.high)
