import math

import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, causal=True):
        super().__init__()
        self.causal = causal
        torch.set_float32_matmul_precision("highest")

    def forward(self, q, k, v):
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if self.causal:
            mask = torch.ones((q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def get_inputs():
    return tuple(make_inputs("flash_attention_fwd"))


def get_init_inputs():
    return [True]
