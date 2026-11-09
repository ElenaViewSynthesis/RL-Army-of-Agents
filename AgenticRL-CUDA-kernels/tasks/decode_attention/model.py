import math

import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_float32_matmul_precision("highest")

    def forward(self, q, k, v):
        repeats = q.shape[1] // k.shape[1]
        expanded_k = k.float().repeat_interleave(repeats, dim=1)
        expanded_v = v.float().repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q.float(), expanded_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        return torch.matmul(torch.softmax(scores, dim=-1), expanded_v).to(q.dtype)


def get_inputs():
    return make_inputs("decode_attention")


def get_init_inputs():
    return []
