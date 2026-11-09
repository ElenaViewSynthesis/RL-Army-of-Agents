import torch
import torch.nn as nn

from utils.task_catalog import active_case
from utils.task_inputs import make_inputs


class Model(nn.Module):
    def __init__(self, top_k=32, top_p=0.9):
        super().__init__()
        self.top_k, self.top_p = top_k, top_p

    def forward(self, logits, uniforms):
        indices = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)[:, :self.top_k]
        values = logits.float().gather(1, indices)
        probabilities = torch.softmax(values, dim=1)
        cumulative = probabilities.cumsum(dim=1)
        retained = cumulative - probabilities < self.top_p
        filtered = torch.where(retained, probabilities, 0.0)
        cdf = (filtered / filtered.sum(dim=1, keepdim=True)).cumsum(dim=1)
        rank = (cdf <= uniforms[:, None]).sum(dim=1).clamp_max(self.top_k - 1)
        return indices.gather(1, rank[:, None]).squeeze(1).to(torch.int32)


def get_inputs():
    return make_inputs("topk_sampling")


def get_init_inputs():
    p = active_case("topk_sampling").params
    return [p["top_k"], p["top_p"]]
