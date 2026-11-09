# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from utils.task_inputs import make_inputs
from utils.task_catalog import active_case
from utils.eagle_reference import tree_conv


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, history, weight, bias, parents):
        return tree_conv(x, history, weight, bias, parents, mamba=False)


def get_inputs():
    return make_inputs("eagle_gdn_tree_conv")


def get_init_inputs():
    return []
