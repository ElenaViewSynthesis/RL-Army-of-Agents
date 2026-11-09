import torch.nn as nn
from kernels.triton import log_softmax_gather_forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, logits, token_ids):
        return log_softmax_gather_forward(logits, token_ids)
