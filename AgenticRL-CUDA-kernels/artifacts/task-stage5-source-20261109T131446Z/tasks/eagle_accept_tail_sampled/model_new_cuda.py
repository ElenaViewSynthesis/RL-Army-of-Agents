# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from kernels.cuda.accept_tail_sampled import forward

BACKEND = "cuda"


class ModelNew(nn.Module):
    def forward(self,tokens, parents, target, draft, slots, uniforms, bonus_uniforms):
        return forward(tokens, parents, target, draft, slots, uniforms, bonus_uniforms)
