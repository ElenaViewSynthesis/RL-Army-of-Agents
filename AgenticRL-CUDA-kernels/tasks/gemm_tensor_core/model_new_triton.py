import torch.nn as nn
from kernels.triton import gemm_tensor_core_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, a, b, bias):
        return gemm_tensor_core_forward(a, b, bias)
