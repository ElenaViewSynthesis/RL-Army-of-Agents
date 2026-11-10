# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from kernels.cuda.tree_step import forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores, children, parents):
        return forward(scores, children, parents)
