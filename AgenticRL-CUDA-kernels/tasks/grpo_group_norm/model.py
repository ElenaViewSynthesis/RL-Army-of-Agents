import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, rewards):
        centered = rewards.float() - rewards.float().mean(dim=-1, keepdim=True)
        std = centered.square().mean(dim=-1, keepdim=True).sqrt()
        return centered / (std + self.eps)


def get_inputs():
    return make_inputs("grpo_group_norm")


def get_init_inputs():
    return [1e-6]
