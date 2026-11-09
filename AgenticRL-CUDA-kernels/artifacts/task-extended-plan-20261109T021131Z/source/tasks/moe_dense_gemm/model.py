import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.set_float32_matmul_precision("highest")

    def forward(self, x, gate, up, down, alpha):
        tokens, hidden = x.shape
        experts, _, intermediate = gate.shape
        packed = torch.stack((gate.float(), up.float()), dim=2).permute(1, 0, 2, 3)
        fc1 = torch.matmul(x.float(), packed.reshape(hidden, 2 * experts * intermediate))
        fc1 = fc1.reshape(tokens, experts, 2, intermediate)
        gated = fc1[:, :, 0] * torch.sigmoid(fc1[:, :, 0]) * fc1[:, :, 1]
        activated = (gated * alpha.float()[:, :, None]).to(x.dtype)
        return torch.matmul(activated.float().reshape(tokens, experts * intermediate),
                            down.float().reshape(experts * intermediate, hidden))


def get_inputs():
    return make_inputs("moe_dense_gemm")


def get_init_inputs():
    return []
