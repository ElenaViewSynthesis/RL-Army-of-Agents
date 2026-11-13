# Task: cuTile kernels adapted from NVIDIA TileGym, in difficulty tiers

You are working in `AgenticRL-CUDA-kernels/` (WSL2). Read `AGENTS.md`, `CLAUDE.md`, the binding
Ground rules in `Benchmark_diagnostics_task.md`, `SKILL_CUTILE.md`, `CUTILE_kernels_task.md`
(the first cuTile round, whose rules all still apply), `Architecture-decision/0005-backend-diagnostic-controls.md`,
and the README sections "Opt-in cuTile task candidates" and "Kernel tasks the agent works on" before
writing anything. **Do not commit and do not launch any GPU run**: Claude reviews, commits from the
Windows checkout and launches approved H200 runs.

## Context

- The repo has a progressive task suite (`tasks/<name>/`, catalog in `utils/task_catalog.py`) scored by
  compile → verify → profile → provenance gate → reward. cuTile is an opt-in **stretch backend**:
  nine tasks have cuTile candidates (`rmsnorm`, `row_softmax`, `cispo_clip_weight`, `fused_adamw`,
  `grpo_group_norm`, `log_softmax_gather`, `residual_rmsnorm`, `gemm_div_sum_scale`, `gemm_tensor_core`).
- **Tier 1 of that work passed on the H200** (`cutile-tier1-20261113T014034Z`, 15/15). cuTile device
  symbols are `<authored function name>_Kt…_<type signature>`, so name-based provenance works unchanged.
  Pins: `cuda-tile==1.8.0`, `nvidia-cuda-tileiras==13.3.36`, H200 (sm_90). Nothing cuTile runs locally.

## Source: NVIDIA TileGym (reference material, not a dependency)

- Repository: <https://github.com/NVIDIA/TileGym>. **Pin tag `v1.5.1`** (released 2026-09-22). Do not use
  `v1.6.0` (released 2026-11-09, too new to pin). Record the tag and the commit SHA you read.
- License: the repository is **MIT** (`LICENSE`, Section A; `SPDX-License-Identifier: MIT` on source
  files). Files under `skills/` (and the `.agents/skills/`, `.claude/skills/` symlinks) are also under a
  CC-BY-4.0 + Apache-2.0 "Agent License": **do not copy from `skills/` or `wiki/`**. Kernels under
  `src/tilegym/suites/liger/` are TileGym's cuTile ports of [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
  operators: record Liger-Kernel as the upstream origin of those contracts as well.
- **Do not `pip install tilegym`** and add no dependency: its `requirements.txt` pulls `transformers`,
  `tokenizers`, `pandas`, `matplotlib` and more. Read the source (download the `v1.5.1` tarball to a
  scratch directory) and **port** the kernel code into each task's `kernels/cutile/`, rewritten to this
  repo's rules below. Keep TileGym's SPDX/copyright header and add a provenance line (URL, tag, commit,
  original path) at the top of every ported file.
- Add a `tilegym` entry to the `SOURCES` registry in `utils/task_catalog.py` (next to `kernelbench`,
  `tritonbench`, `flashinfer`, …) with URL, tag, commit and license, and a `liger` entry for the Liger
  origin; reference them from each new task's `sources` tuple.

TileGym backends: only `src/tilegym/ops/cutile` and `src/tilegym/suites/*/cutile` are in scope. **Out of
scope:** `ops/tilecpp` (CUDA Tile C++, a new toolchain and backend), `ops/triton` (needs the
Triton-to-TileIR fork, a new dependency), `ops/cutile_rs` (Rust). Do not add a backend; ask if you think
one is worth proposing.

## Hardware hazards found in TileGym (handle every one)

1. **Not validated on Hopper.** TileGym's README requires CUDA 13.1+ and a Blackwell GPU (Ampere from
   CUDA 13.2); our H200 is sm_90 and is not on its list. Treat every ported kernel as unverified until an
   H200 run passes.
2. **Architecture branches misroute sm_90.** Example: `ops/cutile/matmul.py` checks `(12, x)`, then
   `capability[0] < 9`, and sends everything else, including (9, 0), down its "sm100+ (Blackwell)" path.
   Replace runtime capability branches with **fixed sm_90 choices** and note each one in the task README.
3. **`ByTarget(sm_100=…, default=…)` hints** resolve to `default` on sm_90, which is tuned for nothing in
   particular. Set explicit values you can justify, or remove the hint.
4. **Autotuning:** most kernels call `cuda.tile.tune.exhaustive_search`. **Remove it**: no tuning in
   `forward` or on the timed path (rule from `CUTILE_kernels_task.md`). Pick fixed tile sizes and record why.
5. **CTA clusters / `num_ctas` and `occupancy` hints:** keep `num_ctas=1` unless you have a reason that holds
   on sm_90; record any occupancy hint you keep. Measurement will decide later.
6. **TMA paths:** some kernels (for example `softmax.py`) switch to TMA when capability ≥ 9. Keep only one
   path per kernel and say which.

## Repo rules the ports must satisfy (TileGym code usually violates some)

- **No torch compute in wrappers or `model_new.py`**: allocation, views and metadata only. TileGym wrappers
  often finish reductions in torch (`suites/liger/cutile/grpo_loss.py::_reduce_loss` and
  `_compute_dapo_normalizer`: about 29 host-side torch math calls). Move every reduction into an authored
  kernel, or the provenance gate will count it as torch time.
- **Forward only.** Task contracts are forward passes. Drop backward kernels and `torch.autograd.Function`
  wrappers. Backward variants are a separate, later decision (Tier 5).
- **Authored kernel names:** at least four characters, descriptive, not generic. Prefer
  `<task>_cutile_<role>` like the first round (TileGym's `_grpo_loss_fwd_ct` style is acceptable only if it
  stays distinctive). Do not edit provenance patterns.
- **No library kernels:** no cuBLAS/cuDNN/CUTLASS calls, no `torch.nn.functional` compute.
- **Verification and tolerances are fixed** by each task's contract (`utils/verification.py`,
  `utils/task_verification.py`, `utils/lora_verification.py`). **ASK before changing any tolerance.**
- **Opt-in only:** declare cuTile under `stretch_backends`; default backends, cases, dtypes and suite row
  counts stay unchanged (fingerprint them before and after, as the first round did).
- **New tasks** (Tiers 1, 2 and the new Tier 3 task) need the full task contract from the earlier briefs:
  trusted `model.py` reference with shapes from the catalog, honest Triton **and** CUDA fixtures, honest
  cuTile fixture, Torch-op hack fixture (must verify and fail provenance), wrong-output fixture (must fail
  verification), generated task README with source, license and shape/dtype grid. New tasks need user
  approval of their contracts and shape grids: **write the contracts and stop for review before
  implementing Tier 2 onward** (see the order below).

## Difficulty tiers

Difficulty reflects what the port demands on top of TileGym's code: removing tuning and arch branches,
moving host reductions into kernels, and (for new tasks) writing the Triton and CUDA fixtures too.

### Tier 1: gated activations and rotary embedding (new small tasks, single kernel, forward)

| New task | TileGym source (`v1.5.1`) | Why it is worth having here | Port notes |
|---|---|---|---|
| `silu_and_mul` | `ops/cutile/silu_and_mul.py` (also `swiglu.py`) | The SwiGLU step of every Llama/Qwen MLP and of our `moe_dense_gemm`; input `[..., 2I]` → `[..., I]`. Also the natural host for SiTU-GLU-style variants later (see `TODO.md`). | Remove `exhaustive_search` and the capability branch. bf16 in, fp32 math, bf16 out. |
| `geglu_tanh` | `ops/cutile/activation/geglu.py` | Gemma-style gated GELU. | TileGym uses the **tanh approximation** of GELU (cuTile 1.8.0 has no `erf`). The contract must say so explicitly; it cannot replace the exact-`erf` `layernorm_residual_gelu` contract. |
| `rotary_embedding` | `ops/cutile/rope.py` | RoPE is applied to Q and K in every attention layer of a rollout policy. | Fixed `cos`/`sin` cache as an input; Neox (half-split) layout first; in-place vs out-of-place must match the reference. |
| `relu_squared` | `suites/liger/cutile/relu_squared.py` | Cheap, exact; a good first port to settle the porting conventions. | Elementwise; trivial to verify. |

Expected difficulty: low. Main risks are the conventions (naming, no tuning, provenance header).

### Tier 2: RL losses over a large vocabulary (new tasks; the differentiator)

These connect directly to the RL objective, like the existing `log_softmax_gather` and
`cispo_clip_weight`. All run a log-sum-exp over a 128,256-wide vocabulary (bf16 logits). **That path is
not yet tested on the H200:** tier 2 of the first cuTile round (`cutile-tier2-20261113T014034Z`, 6/6
passed) ran only the `log_softmax_gather` smoke case, `b32_v1024`, so its 1024-wide online log-sum-exp
loop ran once. The 8,192 and 128,256-wide cases exist only in `--case-set all` (the `acceptance` set is
also `b32_v1024`). Do not finalise the vocabulary tiling until a cuTile `log_softmax_gather` run over the
full grid has passed.

| New task | TileGym source | Why | Port notes |
|---|---|---|---|
| `token_kl_div` | `suites/liger/cutile/kl_div.py` | The KL penalty term in GRPO-style losses (`utils/config.py` GRPO preset has `kl_coef=0.001`). | Inputs are log-probabilities; state the KL direction and reduction in the contract. |
| `generalized_jsd` | `suites/liger/cutile/jsd.py` | Distillation and policy-divergence diagnostics. | `beta` as a compile-time constant; `jsd.py` uses `ByTarget(sm_100=4)`: set an sm_90 value. |
| `fused_cross_entropy` | `suites/liger/cutile/cross_entropy.py` | The SFT and teacher-forcing loss; online log-sum-exp plus target gather. | Has a capability branch and tuning; about 20 host-side torch calls to move into kernels. Label smoothing off in v1. |
| `grpo_token_loss` | `suites/liger/cutile/grpo_loss.py` (token-level path only) | One kernel computing log-sum-exp, target log-prob, importance ratio and the per-token GRPO/DAPO/CISPO loss: the fusion of our `log_softmax_gather` and `cispo_clip_weight`. | Take the **forward, token-level** kernel only (drop backward and the sequence-level GSPO path). Clip bounds and loss type must come from this repo's presets (`clip_low`, `clip_high`, CISPO `[0, 5]`) as compile-time constants. Move `_reduce_loss` and `_compute_dapo_normalizer` into kernels or define the task output as the per-token loss plus mask (decide in the contract and say why). |

Expected difficulty: medium to high. Risks: online log-sum-exp error at V = 128,256, compile-time
branching across loss types, host reductions, and numerics that must match `utils/grpo.py` and the CISPO
definition in `CISPO_GPU_batch_ranking.md` §4.

### Tier 3: GEMM family and MoE (cuTile for existing tasks, plus one new task)

| Task | TileGym source | New or existing | Port notes |
|---|---|---|---|
| `grouped_gemm` | `ops/cutile/group_gemm.py` | existing (Triton-only today) | The task's inputs are lists; check the first round's note that it is "Triton-only/list inputs" and propose how cuTile handles them before porting. |
| `moe_dense_gemm` | `ops/cutile/linear_gluact_linear.py`, `ops/cutile/moe.py` | existing | Dense all-expert FC1, fp32 SwiGLU and alpha mask, dense FC2: match the catalog contract exactly. Ignore `moe.py`'s fp8 paths. |
| `moe_align_block` | `ops/cutile/moe_align_block.py` | **new** | Token → expert sorting into padded blocks (vLLM-style), the MoE analogue of `lora_segment_metadata`. Integer outputs, compared exactly. No tuning or arch checks in the source: the cleanest port in this tier. |
| `gemm_tensor_core` (second cuTile solution, optional) | `ops/cutile/matmul.py` | existing | Only if the first round's simple MMA kernel fails the flagship 60%-of-cuBLAS gate. Its arch branch sends sm_90 to the Blackwell path: fix the tiles for sm_90. |

Expected difficulty: high. Risks: tile choices on sm_90 without the Blackwell tuning, and
`grouped_gemm`'s list inputs.

### Tier 4: attention (cuTile for existing tasks)

| Existing task | TileGym source | Port notes |
|---|---|---|
| `flash_attention_fwd` | `ops/cutile/attention.py` (forward, causal, fixed head dim) | 1,369 lines with backward, variants, tuning and arch branches: extract only the causal forward used by the task contract. |
| `decode_attention` | `ops/cutile/flash_decode.py`, `ops/cutile/gemma_attention_decode.py` (GQA) | Match the task's head counts and KV lengths; split-KV only if the contract allows it. |

Expected difficulty: very high. Risk: register pressure and online-softmax accuracy without TileGym's
Blackwell tuning. Start only after Tiers 1–3 pass on the H200.

### Tier 5: research and blocked (write findings only; implement nothing without approval)

- **Gated delta rule** (`ops/cutile/recurrent_gated_delta_rule.py`, `chunk_gated_delta_rule.py`): related to
  our EAGLE-4 `eagle_gdn_tree_recurrence`. Compare the math and say whether it could serve as a cuTile
  reference or only as a study guide.
- **Mamba2** (`ops/cutile/mamba2.py`, 1,987 lines): related to `eagle_mamba2_tree_conv`. Same question.
- **FP8 block-scaled GEMM** (`ops/cutile/fp8_quantization_matmul.py`, W8A8 e4m3): Hopper has FP8 MMA, so
  this could give our contract-only `fp8_gemm` task its first backend. That changes a contract-only task
  and needs the user's decision.
- **Backward kernels** for the Tier 1–2 tasks: our contracts are forward-only; propose, do not implement.
- **Blackwell-only:** `experimental/nvfp4_quantize.py`, `experimental/sparse_mla.py`, `experimental/mhc.py`:
  excluded (sm_90 cannot run them).

**Excluded with reasons:** `dropout` (random masks cannot match a deterministic reference), `transpose`
(memory-only), `attention_sink*`, `mla*`, `gemma_attention` prefill, `attention_variant`,
`fused_neighborhood_attention`, `multi_token_attention` (model-specific attention variants with no
catalog task; `multi_token_attention` also calls torch), `suites/flashinfer` BSR attention and ragged and
block-scaled BMMs (very large, Blackwell-tuned), `suites/unsloth` duplicates of the above, `layer_norm*`
and the Liger norms (our catalog already covers RMSNorm, residual RMSNorm and LayerNorm+GELU).

## Order of work and stop points

1. **Survey and contracts (no kernels yet).** For every Tier 1–3 item: confirm the TileGym source at
   `v1.5.1`, list each hazard from the sections above that applies, and write the proposed contract for
   each **new** task (reference math, shapes and dtypes for smoke/acceptance/stratified, outputs, tolerance
   class, sources). Put this in `HANDOFF_CODEX_TILEGYM.md` (gitignored). **Stop and report** so the user can
   approve the new tasks and grids.
2. **Tier 1** after approval: implement, wire, test locally (both suites), freeze, and prepare one H200 smoke
   request (`--backends cutile,triton,cuda`, `--fixtures all`). Stop.
3. **Tier 2**, then **Tier 3**, each with its own freeze and request, after the previous tier passes on the
   H200 (Claude launches and reports).
4. **Tier 4** only after Tiers 1–3 pass. **Tier 5**: findings only.

Each tier is one evidence bundle under `artifacts/runs/tilegym-local-<UTC>/` (JSON, logs and a README; no
source copies) and one frozen source per request under `artifacts/sources/tilegym/`, exactly like
`artifacts/runs/cutile-local-20261113T014034Z/`.

## Tests (no GPU needed)

Reuse and extend `tests/cutile_source_support.py`: layout, lazy exports, kernel discovery and naming,
launch on the current stream, fresh outputs, no host-side torch compute, no `exhaustive_search` import,
no runtime capability branch, provenance header present. Default backend and case fingerprints
unchanged for all existing tasks. New tasks: reference shapes, fixture layout, catalog wiring, and the
Torch-op hack and wrong fixtures present.

## Report back

For each tier: files added, the TileGym file and line ranges each kernel came from, every hazard and how
it was handled, fixed tile choices and why, what was removed (tuning, backward, variants, host
reductions), both local test summaries, the prepared Modal commands with row counts and estimates, and
every open question. Nothing committed, nothing launched.
