# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from utils.task_inputs import make_inputs
from utils.task_catalog import active_case
from utils.eagle_reference import candidate_topk


class Model(nn.Module):
    def __init__(self, top_k=8):
        super().__init__()
        self.top_k=top_k

    def forward(self, hidden, weight):
        return candidate_topk(hidden, weight, self.top_k)


def get_inputs():
    return make_inputs("eagle_candidate_topk")


def get_init_inputs():
    return [active_case("eagle_candidate_topk").params["top_k"]]
