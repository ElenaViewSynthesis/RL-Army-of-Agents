import torch.nn as nn
from kernels.cuda import log_softmax_gather_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self, logits, token_ids):
        return log_softmax_gather_forward(logits, token_ids)
