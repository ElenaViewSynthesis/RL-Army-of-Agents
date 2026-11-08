"""cuTile example: copy this file to model_new.py in a task workdir to select it."""

import torch.nn as nn
from kernels.cutile import axpby_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return axpby_forward(a, b, self.alpha)
