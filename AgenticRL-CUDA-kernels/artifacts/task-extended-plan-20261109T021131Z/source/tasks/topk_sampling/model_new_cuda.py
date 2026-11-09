import torch.nn as nn
from kernels.cuda import topk_sampling_forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def __init__(self, top_k=32, top_p=0.9):
        super().__init__()
        self.top_k, self.top_p = top_k, top_p

    def forward(self, logits, uniforms):
        return topk_sampling_forward(logits, uniforms, self.top_k, self.top_p)
