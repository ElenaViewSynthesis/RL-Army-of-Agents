# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
from model import Model

BACKEND = "triton"


class ModelNew(Model):
    def forward(self,*args):
        result=list(super().forward(*args))
        # Deliberate off-by-one parent/ancestor metadata; verification must fail.
        result[4]=result[4]+1
        return tuple(result)
