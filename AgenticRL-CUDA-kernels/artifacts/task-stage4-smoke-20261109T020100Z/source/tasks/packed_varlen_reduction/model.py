import torch
import torch.nn as nn

from utils.task_catalog import active_case
from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, values, offsets):
        sums = torch.segment_reduce(values.float(), "sum", offsets=offsets)
        if self.reduction == "sum":
            return sums
        lengths = offsets[1:] - offsets[:-1]
        return sums / lengths.clamp_min(1).float()


def get_inputs():
    return make_inputs("packed_varlen_reduction")


def get_init_inputs():
    return [active_case("packed_varlen_reduction").params["reduction"]]
