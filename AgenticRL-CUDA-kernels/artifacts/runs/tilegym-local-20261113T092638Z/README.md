# TileGym `v1.5.1` source survey and contract proposals

Local evidence for step 1 of `TILEGYM_kernels_task.md`: a survey of NVIDIA TileGym's cuTile sources and
proposed contracts for nine new tasks. **Nothing was implemented, no catalog or test changed, and no
GPU run was launched.** TileGym was read, not installed and not executed. This folder holds no copy of
its source.

Pin: tag `v1.5.1`, commit `bb1a7e720594880be017e8fdf8f7e4f740656beb`, archive
`https://codeload.github.com/NVIDIA/TileGym/tar.gz/refs/tags/v1.5.1`, SHA-256
`17e769753731efe126829672fdc51a1a6506ac3d13a3879e51c278586cb720b6`.

## Files

| File | Written by | Contents |
|---|---|---|
| `source-survey.json` | Codex | For 18 surveyed files: SHA-256, line count, the line range of every function and class, and the line numbers of 11 hazard keywords (`exhaustive_search`, `get_device_capability`, `ByTarget`, `num_ctas`, `occupancy`, `check_bounds=False`, `allow_tma`, `torch.autograd`, `torch.sum`, `.item()`, `fill_`). |
| `proposed-sources.json` | Codex | Proposed `SOURCES` entries: `tilegym` (MIT) and `liger` (BSD-2-Clause). Not added to the catalog. |
| `repository-check.json` | Codex | Catalog fingerprints at survey time: 35 tasks, case and default-backend hashes, protected files unchanged. |
| `contracts-summary.json` | Claude | For each of the nine proposed tasks: tier, reference formula, inputs, outputs, verifier, sources with line ranges, hazards and how each is resolved, fixed tile choices, and the full case list with logical byte counts. Also the three existing Tier 3 tasks and whether a port needs a contract change. |
| `reference-checks.json` | Claude | CPU checks of the proposed reference math on synthetic inputs. |

## Verification of the survey (2026-11-13)

The archive was downloaded again and its SHA-256 matches. All 18 file hashes and line counts match. All
163 function and class ranges were compared with Python's `ast`: every end line is exact; 100 ranges
start at the `def` line and 63 at the first decorator line. For every file and keyword, the listed
hazard lines equal the lines that contain the keyword. The keyword list is narrow: host calls such as
`torch.gather` (`cross_entropy.py:427`) are not in it.

## What `contracts-summary.json` is and is not

It is a proposal awaiting the user's approval. Case IDs follow `utils.task_catalog.case`. Byte counts
are formulas over the proposed shapes (`size_note`), not measured memory. `last_case_has_largest_shape`
is true for every task; `last_case_has_most_bytes` is false for the six tasks whose last case is bf16,
because the fp32 case of the same shape has twice the bytes.

## What `reference-checks.json` is and is not

Torch 2.14.1+cpu in the WSL `.venv`, seed 0, synthetic random inputs, tolerance atol = rtol = 1e-2
(the fixed tolerance of `utils/verification.py`). It checks conventions of the proposed references. It
is **not GPU evidence** and tests no kernel; `gpu_metrics` is `null`. Findings:

- Tanh-approximate GELU differs from exact `erf` GELU by at most 4.7 × 10⁻⁴. TileGym's
  `approximate="none"` path equals the tanh path to 1.8 × 10⁻¹⁵ in float64: it is not exact.
- Emulated bf16 arithmetic, as TileGym's GEGLU and RoPE kernels use, leaves elements outside the
  tolerance (GEGLU: 1 of 4,194,304; RoPE: 1.3 × 10⁻⁵ of elements). The ports need fp32 arithmetic.
- A CISPO cap of 4 or 6 instead of 5 fails verification on the per-token output and on the scalar.
- Dropping the GRPO KL term at `kl_coef = 0.001` stays inside the tolerance; at 0.04 with a wider
  reference spread it fails on 53% of tokens.
- Swapping student and teacher in the generalised JSD stays inside the tolerance, also at β = 0.1.
  Swapping the KL direction does not.
- The proposed `moe_align_block` reference reproduces the worked example in TileGym's docstring.

The full proposal, with the reasoning and the open decisions, is in the gitignored working note
`HANDOFF_CODEX_TILEGYM.md` at the root of `AgenticRL-CUDA-kernels/`.
