import torch.nn as nn
from kernels.triton import grouped_gemm_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, matrices_a, matrices_b):
        return grouped_gemm_forward(matrices_a, matrices_b)
