import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    """x [B,2H] -> silu(x[:, :H]) * x[:, H:]; SiLU on the FIRST half, fp32 math, input-dtype output."""

    def forward(self, x):
        h = x.shape[-1] // 2
        gate, up = x[:, :h].float(), x[:, h:].float()
        return (gate * torch.sigmoid(gate) * up).to(x.dtype)


def get_inputs():
    return make_inputs("silu_and_mul")


def get_init_inputs():
    return []
