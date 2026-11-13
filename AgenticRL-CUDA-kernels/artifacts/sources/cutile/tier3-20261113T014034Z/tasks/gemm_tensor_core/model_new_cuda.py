import torch.nn as nn
from kernels.cuda import gemm_tensor_core_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, a, b, bias):
        return gemm_tensor_core_forward(a, b, bias)
