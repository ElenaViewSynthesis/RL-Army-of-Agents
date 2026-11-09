import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def forward(self, x, lengths):
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        sums = torch.where(mask, x.float(), 0.0).sum(dim=1)
        return sums / lengths.clamp_min(1).float()


def get_inputs():
    return make_inputs("masked_mean")


def get_init_inputs():
    return []
