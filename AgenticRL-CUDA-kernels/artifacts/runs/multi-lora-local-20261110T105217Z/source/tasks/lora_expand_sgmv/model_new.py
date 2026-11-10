import torch.nn as nn
from kernels.triton.lora import forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return forward('lora_expand_sgmv', *args)
