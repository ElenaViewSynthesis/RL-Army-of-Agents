import torch.nn as nn
from kernels.cuda.lora import forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return forward('lora_shrink_sgmv', *args)
