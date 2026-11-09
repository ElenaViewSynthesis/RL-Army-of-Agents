import torch.nn as nn
from kernels.triton import decode_attention_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, q, k, v):
        return decode_attention_forward(q, k, v)
