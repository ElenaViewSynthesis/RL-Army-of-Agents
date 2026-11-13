import torch.nn as nn
from kernels.cutile import fused_adamw_forward

BACKEND = "cutile"


class ModelNew(nn.Module):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=10):
        super().__init__()
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.eps, self.weight_decay, self.step = eps, weight_decay, step

    def forward(self, parameter, gradient, first_moment, second_moment):
        return fused_adamw_forward(parameter, gradient, first_moment, second_moment,
                                   self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.step)
