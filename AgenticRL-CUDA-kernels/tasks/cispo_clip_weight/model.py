import torch
import torch.nn as nn

from utils.config import RolloutConfig
from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, low=0.0, high=5.0):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, log_ratio, advantage, mask):
        weight = log_ratio.float().exp().clamp(self.low, self.high).detach()
        return torch.where(mask, weight * advantage.float()[:, None], 0.0)


def get_inputs():
    return make_inputs("cispo_clip_weight")


def get_init_inputs():
    cfg = RolloutConfig.preset("cispo")
    return [1 - cfg.clip_low, 1 + cfg.clip_high]
