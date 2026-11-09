# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
def forward(x,history,weight,bias,parents,mamba=False):
    import cuda_extension
    return tuple(cuda_extension.eagle_forward([x,history,weight,bias,parents],int(mamba)))
