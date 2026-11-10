"""CuTe DSL example: copy this file to model_new.py in a task workdir to select it."""

import torch.nn as nn
from kernels.cutedsl import axpby_forward

BACKEND = "cutedsl"


class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return axpby_forward(a, b, self.alpha)
