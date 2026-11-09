import torch.nn as nn
from kernels.cuda import flash_attention_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def __init__(self, causal=True):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return flash_attention_forward(q, k, v, self.causal)
