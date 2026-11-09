import torch.nn as nn
from kernels.triton import gemm_div_sum_scale_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def __init__(self, divisor=2.0, scale=1.5):
        super().__init__()
        self.divisor, self.scale = divisor, scale

    def forward(self, a, b):
        return gemm_div_sum_scale_forward(a, b, self.divisor, self.scale)
