# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from utils.task_inputs import make_inputs
from utils.task_catalog import active_case
from utils.eagle_reference import attention


class Model(nn.Module):
    def __init__(self, window=512):
        super().__init__()
        self.window=window

    def forward(self, q, k, v, tk, tv, parents):
        return attention(q, k, v, tk, tv, parents, self.window)


def get_inputs():
    return make_inputs("eagle_tree_row_attention")


def get_init_inputs():
    return [active_case("eagle_tree_row_attention").params["window"]]
