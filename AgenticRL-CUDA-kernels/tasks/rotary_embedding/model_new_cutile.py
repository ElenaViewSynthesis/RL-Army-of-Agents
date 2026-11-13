import torch.nn as nn
from kernels.cutile import rotary_embedding_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def forward(self, q, k, cos, sin):
        return rotary_embedding_forward(q, k, cos, sin)
