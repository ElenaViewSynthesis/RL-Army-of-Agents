import torch.nn as nn
from kernels.cuda import decode_attention_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, q, k, v):
        return decode_attention_forward(q, k, v)
