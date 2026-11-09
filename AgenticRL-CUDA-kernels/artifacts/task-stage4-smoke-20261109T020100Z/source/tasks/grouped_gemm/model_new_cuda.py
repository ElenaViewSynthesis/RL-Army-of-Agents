import torch.nn as nn

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, matrices_a, matrices_b):
        raise RuntimeError("unavailable: grouped_gemm is Triton-only for now")
