# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from kernels.triton.candidate_topk import forward

BACKEND = "triton"


class ModelNew(nn.Module):
    def __init__(self, top_k=8):
        super().__init__()
        self.top_k=top_k

    def forward(self, hidden, weight):
        return forward(hidden, weight, self.top_k)
