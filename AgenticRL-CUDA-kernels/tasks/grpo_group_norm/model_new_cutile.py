import torch.nn as nn
from kernels.cutile import grpo_group_norm_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, rewards):
        return grpo_group_norm_forward(rewards, self.eps)
