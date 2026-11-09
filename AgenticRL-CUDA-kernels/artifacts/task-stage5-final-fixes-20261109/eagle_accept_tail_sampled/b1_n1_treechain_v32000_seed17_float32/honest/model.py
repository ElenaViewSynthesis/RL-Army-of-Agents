# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch.nn as nn
from utils.task_inputs import make_inputs
from utils.eagle_reference import accept_sampled


class Model(nn.Module):
    def forward(self,tokens, parents, target, draft, slots, uniforms, bonus_uniforms):
        return accept_sampled(tokens, parents, target, draft, slots, uniforms, bonus_uniforms)


def get_inputs():
    return make_inputs("eagle_accept_tail_sampled")


def get_init_inputs():
    return []
