import torch
import torch.nn as nn

from utils.task_inputs import make_inputs


class Model(nn.Module):
    def forward(self, logits, token_ids):
        values = torch.log_softmax(logits.float(), dim=-1)
        return values.gather(1, token_ids[:, None]).squeeze(1)


def get_inputs():
    return make_inputs("log_softmax_gather")


def get_init_inputs():
    return []
