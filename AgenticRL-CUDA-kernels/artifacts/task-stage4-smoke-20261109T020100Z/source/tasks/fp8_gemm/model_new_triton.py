import torch.nn as nn

BACKEND = "triton"


class ModelNew(nn.Module):
    def forward(self, a, b, scale_a, scale_b):
        raise RuntimeError("unavailable: contract-only; no authored quantized kernel shipped")
