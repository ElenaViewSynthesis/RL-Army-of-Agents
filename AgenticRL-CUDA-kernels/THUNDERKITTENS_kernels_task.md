# Task collection: kernels from HazyResearch ThunderKittens, for the H200

A proposed collection of tasks drawn from [ThunderKittens](https://github.com/HazyResearch/ThunderKittens),
in difficulty tiers. **Nothing here is implemented.** It follows the same two-step process as
[`TILEGYM_kernels_task.md`](TILEGYM_kernels_task.md): first a source survey and contract proposal,
then a stop for the user's approval before any kernel is written.

Read `AGENTS.md`, `CLAUDE.md`, the Ground rules in `Benchmark_diagnostics_task.md`, `SKILL_CUDA.md`
and `TILEGYM_kernels_task.md` (whose porting rules all apply here) before acting. **Do not commit and
do not launch any GPU run** without the user's approval.

## Source: what ThunderKittens is

Facts below come from the repository's README and directory listings, read on 2026-11-13. They are not
yet pinned to a commit: step 1 must record the commit SHA it reads, as the TileGym survey did.

- **What it is:** a framework for writing fast deep-learning kernels in CUDA C++. It is a
  header-only library (include `kittens.cuh`) of tile types (register and shared-memory tiles and
  vectors) and operations on them.
- **Hardware:** "mainly built and tested for Hopper and Blackwell GPUs". Kernels come in per-GPU
  folders; the `*_h100` ones target sm_90, which is the architecture of this repo's H200. Ampere is
  no longer actively supported.
- **Toolchain:** CUDA 12.8 or newer and C++20. Kernels are exposed to PyTorch through PyBind11
  extensions, each compiled from its own directory.
- **License:** MIT.
- **Papers:** [arXiv:2410.20399](https://arxiv.org/abs/2410.20399) (single GPU) and
  [arXiv:2511.13940](https://arxiv.org/abs/2511.13940) (multiple GPUs).
- **Attention:** the README presents its attention kernel as "Flash Attention 3, but with kittens!",
  a FlashAttention-3-style kernel built from the library's tiles. The README gives no measured
  comparison with FlashAttention-3.
- **Kernel folders** (`kernels/`): `attention`, `based`, `fftconv`, `flux`, `gemm`, `hedgehog`,
  `layernorm`, `linear_attention`, `mamba2`, `parallel`, `rotary`. Inside `attention`:
  `mha_h100`, `mha_h100_lcf`, `bf16_b300_mha_causal`, `bf16_b300_mha_noncausal` (the listing was
  truncated). Inside `gemm`: `bf16_h100`, `fp8_h100`, `fp8_h100_scaled`, `int8_h100`,
  `educational_h100`, plus Blackwell and Vera Rubin variants (`*_b200`, `*_b300`, `*_vr200`,
  `mxfp8_*`, `nvfp4_*`).

## The decision to make first: how ThunderKittens may be used

ThunderKittens is a library of prewritten tile primitives (MMA wrappers, TMA loads, warp-group
scheduling). This repo's CUDA rules forbid "third-party kernels or prewritten compute libraries" in a
candidate. Two ways to use it; **the user decides before step 1 ends**:

| Option | What it means | Cost |
|---|---|---|
| **A. Reference only (recommended first)** | Read ThunderKittens' sm_90 kernels and port their techniques into plain CUDA C++ fixtures. No ThunderKittens header is compiled. | Harder fixtures; no dependency or policy change. |
| **B. Headers as a vendored dependency** | Allow `kittens.cuh` in a CUDA variant (for example a `cuda-tk` stretch backend). Authored `__global__` names would still pass the name-based provenance gate, but the kernel's inner loops would be library code. | A new dependency (needs approval), a C++20 build flag (`utils/compile.py` passes only `-O3 --use_fast_math` today, unverified whether nvcc defaults to C++20), and a change to what "authored" means. |

## Tiers

### Tier 1: existing tasks that have a ThunderKittens sm_90 counterpart

| Existing task | ThunderKittens source | What the port would add | Notes |
|---|---|---|---|
| `flash_attention_fwd` | `kernels/attention/mha_h100` (and `mha_h100_lcf`) | A Hopper-tuned (FA3-style) causal forward as a second CUDA solution | The task's contract is causal, `d = 64`, bf16, `S` up to 2,048. Check which head dimensions and masks the kernel supports at the pinned commit. |
| `gemm_tensor_core` | `kernels/gemm/bf16_h100` (`educational_h100` as a study guide) | A warp-specialized, TMA-fed bf16 GEMM against the 60%-of-cuBLAS gate | The README cites about 855 TFLOP/s on an H100 for its matmul example; not measured here. |
| `rotary_embedding` (Stage 7) | `kernels/rotary` | A third reference layout to compare with TileGym's | Check whether it rotates in place; the Stage 7 contract requires new output tensors. |
| `layernorm_residual_gelu` | `kernels/layernorm` | Partial match only | Check what the kernel fuses; this task's contract also needs exact-`erf` GELU. |

### Tier 2: new tasks or new backends for contract-only tasks

| Proposed task | ThunderKittens source | Why | Decision needed |
|---|---|---|---|
| `flash_attention_bwd` (new) | `kernels/attention/mha_h100`, if it contains the backward pass (to verify) | Training needs the backward; every current task is forward-only | A new forward-plus-gradient contract and its verifier |
| `fp8_gemm` backend | `kernels/gemm/fp8_h100`, `fp8_h100_scaled` | The contract-only `fp8_gemm` task has no backend; Hopper has FP8 MMA | Turns a contract-only task into a measured one (same open question as TileGym Tier 5) |
| `int8_gemm` backend | `kernels/gemm/int8_h100` | Same, for the contract-only `int8_gemm` | Same |

### Tier 3: linear attention and state-space kernels (research)

| Proposed task | ThunderKittens source | Relation to this repo |
|---|---|---|
| `linear_attention_fwd` | `kernels/linear_attention`, `kernels/hedgehog` | No catalog task yet; LoLCATS-style linear attention is used in the README's demos |
| `based_attention_fwd` | `kernels/based` | The Based architecture's forward pass |
| `mamba2_chunk_fwd` | `kernels/mamba2` | Related to Stage 5's `eagle_mamba2_tree_conv`; compare the math before proposing a contract |
| `fftconv_fwd` | `kernels/fftconv` | Long convolutions by FFT; no catalog task yet |

### Excluded, with reasons

- Blackwell and Vera Rubin kernels (`*_b200`, `*_b300`, `*_vr200`, `mxfp8_*`, `nvfp4_*`): sm_90
  cannot run them.
- `kernels/parallel`: multi-GPU; the environment runs one GPU per job.
- `kernels/flux`: not reviewed; classify it in step 1.

## Order of work and stop points

1. **Survey and contracts, no kernels.** Pin a commit; record its SHA, the license and every file
   read. For each Tier 1–3 item: confirm the source, list hazards (architecture branches, autotuning,
   in-place writes, host-side torch math, missing backward), and write the contract for each new
   task (reference math, shapes for smoke, acceptance as the largest case, and the full grid,
   verifier class, sources). Record the user's choice between options A and B. Put this in a
   gitignored `HANDOFF_CODEX_THUNDERKITTENS.md` and **stop for approval.**
2. **Tier 1** after approval: implement, test locally (both suites), freeze, prepare one H200 smoke
   request. Stop.
3. **Tier 2**, then **Tier 3**, each with its own freeze and request, after the previous tier
   passes on the H200.

Each tier records its evidence under `artifacts/runs/` and its frozen source under
`artifacts/sources/thunderkittens/`, as Stage 7 did. Add a `thunderkittens` entry to `SOURCES` in
`utils/task_catalog.py` (MIT) only when the first task that cites it is implemented.
