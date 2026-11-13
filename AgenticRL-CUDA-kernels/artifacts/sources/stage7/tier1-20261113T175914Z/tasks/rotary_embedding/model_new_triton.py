import torch.nn as nn
from kernels.triton import rotary_embedding_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, q, k, cos, sin):
        return rotary_embedding_forward(q, k, cos, sin)
