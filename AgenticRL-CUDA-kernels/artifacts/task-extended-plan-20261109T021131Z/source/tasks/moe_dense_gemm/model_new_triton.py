import torch.nn as nn
from kernels.triton import moe_dense_gemm_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, x, gate, up, down, alpha):
        return moe_dense_gemm_forward(x, gate, up, down, alpha)
