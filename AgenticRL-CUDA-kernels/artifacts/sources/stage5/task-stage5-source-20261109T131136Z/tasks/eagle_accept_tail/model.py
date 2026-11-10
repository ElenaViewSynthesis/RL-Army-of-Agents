# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from utils.task_inputs import make_inputs
from utils.task_catalog import active_case
from utils.eagle_reference import accept


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens, parents, targets, slots):
        return accept(tokens, parents, targets, slots)


def get_inputs():
    return make_inputs("eagle_accept_tail")


def get_init_inputs():
    return []
