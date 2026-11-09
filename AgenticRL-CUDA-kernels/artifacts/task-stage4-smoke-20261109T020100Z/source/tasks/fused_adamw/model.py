import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=10):
        super().__init__()
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.eps, self.weight_decay, self.step = eps, weight_decay, step

    def forward(self, parameter, gradient, first_moment, second_moment):
        m = self.beta1 * first_moment + (1 - self.beta1) * gradient
        v = self.beta2 * second_moment + (1 - self.beta2) * gradient.square()
        m_hat = m / (1 - self.beta1 ** self.step)
        v_hat = v / (1 - self.beta2 ** self.step)
        updated = parameter * (1 - self.lr * self.weight_decay) - self.lr * m_hat / (v_hat.sqrt() + self.eps)
        return updated, m, v


def get_inputs():
    return make_inputs("fused_adamw")


def get_init_inputs():
    return [0.001, 0.9, 0.999, 1e-8, 0.01, 10]
