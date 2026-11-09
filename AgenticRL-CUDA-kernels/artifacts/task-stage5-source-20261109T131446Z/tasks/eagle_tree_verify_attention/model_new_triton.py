# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from kernels.triton.tree_verify_attention import forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def __init__(self, window=512):
        super().__init__()
        self.window=window

    def forward(self, q, k, v, tk, tv, parents):
        return forward(q, k, v, tk, tv, parents, self.window)
