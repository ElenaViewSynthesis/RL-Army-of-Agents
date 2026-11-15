# Refined Prompt: Extending AgenticRL-CUDA-kernels Beyond AXPBY (refreshed 2026-11-15)

This is the progressive task-suite brief with its stale statements brought up to date. The original, as it was
written before Stages 0–4 were built, is kept unchanged in `PROGRESSIVE_TASK_SUITE_task.md`.

**Status as of 2026-11-15:** every task this brief asks for exists, and all except the two contract-only quantized
GEMMs have H200 evidence. The suite has grown past it to 39 tasks in Stages 0–7 (catalog: `utils/task_catalog.py`;
per-stage H200 status: the README's "Kernel tasks the agent works on" section). Use this brief for the conventions and
contracts; the curriculum below records what was built rather than what remains.

## Role

You are a GPU kernel engineer extending the `AgenticRL-CUDA-kernels/` evaluation environment (repo: `ElenaViewSynthesis/RL-Army-of-Agents`, branch `main`). The root workdir still ships the element-wise `alpha*a + b` example on `(1,128)` fp32 across five backends (CUDA, Triton, TileLang, CuTe DSL, cuTile), and it stays the protected default task. Alongside it, `tasks/` now holds a **progressive task suite** of 39 kernel workloads, each conforming to the existing task contract, so the RL candidate-evaluation pipeline has non-trivial kernels to optimize. New work extends that suite.

## Hard constraints (from the existing harness — do not violate)

- Each task is a workdir shaped like the root: a protected `model.py` (PyTorch reference + `get_inputs()`), `binding.cpp`, `binding_registry.h`, `kernels/`, and a candidate `model_new.py`. Root `model.py` stays untouched; new tasks go under `tasks/<name>/` per `Benchmark_diagnostics_task.md` Step 5. Shapes and dtypes come from `utils/task_catalog.py` (`select_cases(task, "smoke"|"acceptance"|"stratified"|"all")`), so a candidate cannot redefine them.
- `ModelNew` must match the reference constructor and `state_dict`; `forward` must enqueue all device work on `torch.cuda.current_stream()`.
- No `torch.*` compute ops in `model_new.py` or wrappers — allocation and metadata only. Compute belongs in authored kernels (`@triton.jit`, `@cute.kernel`, `__global__`, `T.prim_func`, `@ct.kernel`).
- No prewritten library kernels: no `triton.ops`, flash-attn packages, cuBLAS/cuDNN/CUTLASS-library calls. `tl.dot` is allowed; CuTe DSL GEMMs must be built from MMA/copy atoms, not `cutlass.utils` templates.
- Verification: 5 random inputs, atol/rtol = 1e-2 for floating outputs. Integer outputs (scans, IDs, ancestor tables, token indices) are compared exactly, and the Stage 6 multi-LoRA tasks use an approved relative-error bound. **ASK before changing tolerances** for any task.
- Provenance: the task suite applies `authored-only-no-library-v1`, which requires ≥90% of non-memory device time attributed to authored kernel names. Keep default symbol naming.
- Every task needs three fixture variants: honest implementation, a Torch-op reward-hack variant (`model_new_hack.py`: must pass verification, fail provenance), and a wrong-output variant (`model_new_wrong.py`), mirroring `tests/support.py:make_task`/`make_spoof_task`. Tasks with structure to get wrong add more controls (for example `wrong_ancestor` in Stage 5, `wrong_padding` and `wrong_skip` in Stage 6).
- Results stay flat and JSON-safe; result keys are added, never renamed or redefined. No new Python dependencies without the user's approval: the base install is `torch>=2.3`, `triton>=3.0` (`requirements.txt`); the cloud stack is pinned in `requirements-gpu.txt` (`torch==2.14.1+cu130`, `nvidia-cutlass-dsl[cu13]==4.8.1`, `cuda-tile==1.8.0`, `tilelang==0.1.15`), with per-backend files for TileLang, CuTe DSL and cuTile.
- GPU work runs on Modal: the H200 is the default (`cloud/modal_tasks.py`, CUDA 13.3.1 image, `sm_90`), and a B200 (CUDA 13.5.0 image, `sm_100`) is used for Blackwell profiling (`cloud/modal_profile.py`, `cloud/modal_nsight_perf.py`). Every paid run needs the user's approval and runs detached from a frozen source (`scripts/freeze_task_suite.py`). The dev host has no NVIDIA GPU. Everything must degrade to explicit "unavailable" on CPU.

## Source benchmark suites and sample tasks to extract

For each suite below, port the *task definition* (operator contract, shape grid, dtype set, correctness rule) — not their harness code. Record provenance of each task (source suite + URL + license) in the task's README. The task names in brackets are the ones now in the catalog.

### 1. KernelBench (highest priority — task-format model)

License: MIT-style research. Extract 3 samples:

- **Level 1 — `fused_linear_scale` / `gemm_div_sum_scale`** [`gemm_div_sum_scale`, Stage 0, built and passing on the H200]: `C = (A @ B) / s` then column-sum-scale epilogue. Shapes: `M,N,K ∈ {256, 512, 1024}`, fp32 + bf16 (54 catalog cases).
- **Level 2 — fused reduction chain** [`layernorm_residual_gelu`, Stage 2]: `x → LayerNorm → residual add → GELU` (exact erf GELU). Teaches multi-op fusion within one kernel launch.
- **Level 3 — attention block**: full `softmax(QK^T/√d)V` as the reference `model.py` — became the FlashAttention task below.

### 2. GPU MODE reference kernels (highest priority — rigor model)

License: research-reciprocity — **review terms before using problems as training data**; the tasks derived from it are marked eval-only in the catalog. Extract 3 samples:

- **Trimmed mean / masked reduction** [`masked_mean`, Stage 1, eval-only]: row-wise mean over variable-length rows with a length tensor — directly relevant to RL sequence masking. Shapes: `B ∈ {64, 512}`, `L ∈ {128..8192}` ragged.
- **Sort/scan** [`prefix_sum`, Stage 1, eval-only]: block-level prefix sum (Blelloch or warp-scan), `N ∈ {1e5..1e7}` — exercises shared memory and inter-warp sync. Integer outputs, compared exactly.
- **INT8/FP8 quantized GEMM** [`int8_gemm`, `fp8_gemm`, Stage 4]: still contract-only, with no fixtures and no GPU runs. Giving them a backend is the user's decision.

### 3. TritonBench (high priority — backend parity)

Extract 3 samples:

- **`tl.dot` GEMM with epilogue** [`gemm_tensor_core`, Stage 2]: `C = A@B + bias`. This is the **primary Tensor Core task** — `tl.dot` on fp16/bf16 inputs with fp32 accumulation lowers to MMA on H200. Shape grid: square `{512,1024,2048,4096}` plus skinny `M ∈ {8,64}` × `N=K=4096` (RL-inference-like). The honest Triton fixture autotunes over a fixed config list; newer ports (cuTile, TileGym) use fixed tile sizes instead, because autotuning on the timed path changes what is measured.
- **RMSNorm forward** [`rmsnorm`, Stage 2]: row reduction + scale, `H ∈ {1024..8192}`, fp32 accumulate on bf16 input.
- **Grouped GEMM** [`grouped_gemm`, Stage 4]: `G ∈ {4,8}` expert matrices, per-group `M` — MoE-relevant, still Triton-only.

### 4. FlashInfer-Bench (high priority — realistic LLM workloads)

Extract 3 samples:

- **RMSNorm + residual fused** [`residual_rmsnorm`, Stage 4] (prefill shape): `(B*S, H)` with `S` variable.
- **Top-k / top-p sampling kernel** [`topk_sampling`, Stage 4]: logits `(B, V=128256)` → sampled token — rollout-generation-relevant, with deterministic tie rules.
- **Decode attention (single-query)** [`decode_attention`, Stage 4]: `H_q=32, H_kv=8` GQA, `S_kv ∈ {512..8192}` — the decode-side counterpart to FlashAttention.

### 5. CUTLASS profiler (baseline, not a task source)

Its intended role, a reference GEMM ceiling per shape and dtype so results can be reported as "percent of cuBLAS" instead of only speedup-vs-eager, is now filled by `utils/task_ceiling.py`, an opt-in diagnostic that times cuBLAS through PyTorch on the same inputs and is never read by the reward. `cutlass_profiler` itself was not adopted; it would be a new dependency. Candidates still may not call CUTLASS-library kernels.

### 6. NVBench (measurement methodology)

Don't port tasks; its conventions were adopted in `utils/benchmark.py` (opt-in `--benchmark`): rotated inputs, cold vs. warm timing (`--cold-start`), CUDA-graph replay (`--cuda-graphs`), per-call sample sidecars, and achieved-bandwidth fields such as `bench/generated/logical_gbytes_per_s`. Benchmark values are diagnostics and never feed the reward.

### 7. Official Triton tutorials (implementation references)

Use as the *honest fixture* starting points: `fused-softmax`, `matmul` (tl.dot), `layer-norm`, `fused-attention` (the FlashAttention tutorial), `grouped-gemm`, `persistent-matmul`. Rewrite into the repo's `kernels/triton/` style — tutorials are reference material, not copy-paste solutions.

## The two flagship tasks (specify these in full)

### Task A — `tasks/flash_attention_fwd/` (built; Stage 4)

- **Reference `model.py`**: `O = softmax(Q@K^T / sqrt(d)) @ V`, causal flag on, `B=4, H=8, S ∈ {512,1024,2048}, d=64`, bf16 inputs, fp32 softmax accumulation. `get_inputs()` returns `(Q,K,V)`.
- **Honest fixtures**: Triton (online-softmax tiled, from the official tutorial pattern) and CUDA (two-pass shared-memory tiled, acceptable to be slower) both pass smoke and acceptance on the H200. CuTe DSL (MMA atoms + online softmax) and TileLang (`T.gemm` + pipelined softmax) remain stretch backends; a cuTile port is TileGym Tier 4 and waits for Tiers 1–3.
- **Verification**: the tolerance stays at atol/rtol 1e-2 (the catalog contract); the question of relaxing it to 2e-2 for bf16 did not need to be asked because the fixtures pass at 1e-2. Any later change still needs the user's approval.
- **Benchmark**: report TFLOP/s (`4·B·H·S²·d` for causal ≈ half) and % of a recorded cuBLAS/FA2 ceiling.
- **Reward-hack fixture**: `torch.nn.functional.scaled_dot_product_attention` in `model_new.py` — must pass verification, fail provenance (this is exactly the attack the gate exists for).

### Task B — `tasks/gemm_tensor_core/` (built; Stage 2)

- **Reference `model.py`**: `C = A @ B + bias`, `A:(M,K)`, `B:(K,N)` fp16/bf16, `C:(M,N)` fp32, grid `{512,1024,2048,4096}²` plus skinny `M ∈ {8,64}` cases (12 catalog cases).
- **Honest fixtures**: Triton `tl.dot` (fp32 `acc`, `allow_tf32` off for the fp32 variant); CUDA WMMA/`mma.sync` m16n8k16 kernel (the explicit Tensor Core path — shared-memory double-buffered tiles); CuTe DSL via `make_tiled_mma` atoms; cuTile (`ct.mma`) as an opt-in stretch backend.
- **Acceptance (met)**: all three default backends pass smoke and acceptance at 4096³ bf16 on the H200. The honest Triton fixture reaches 85.4% of cuBLAS (gate ≥ 60%, Triton only), provenance = 1.0, and the hack fixture fails provenance. CuTe DSL is correct but slow, at 2.6% of cuBLAS. The cuTile flagship acceptance request (same 60% gate) has not been launched.

## Curriculum ordering (as built)

| Stage | Tasks | New capability exercised | H200 status |
|---|---|---|---|
| 0 — bootstrap | `gemm_div_sum_scale` | GEMM with a fused epilogue | pass |
| 1 — fundamentals | `row_softmax`, `masked_mean` (GPU MODE), `prefix_sum` | reductions, masks, shared memory | pass |
| 2 — training kernels | `gemm_tensor_core` (Task B), `rmsnorm`, `layernorm_residual_gelu`, `log_softmax_gather` (token log-probs), `fused_adamw` | `tl.dot`/MMA, multi-pass fusion | pass |
| 3 — RL-specific | `grpo_group_norm` (per-group mean/std over reward groups), `cispo_clip_weight` (clipped IS weight × advantage, token-level), `packed_varlen_reduction` | the actual GRPO/DAPO/CISPO math as kernels — unique to this repo, no external source needed | pass |
| 4 — advanced | `flash_attention_fwd` (Task A), `decode_attention`, `grouped_gemm`, `moe_dense_gemm`, `residual_rmsnorm`, `topk_sampling`, `fp8_gemm` and `int8_gemm` (contract only) | online softmax, MMA pipelines, MoE | pass, except the two contract-only tasks |

Stage 3 is the differentiator: it connects kernel optimization to the RL objective itself and no external suite covers it — its contracts derive from `utils/grpo.py` and the `utils/config.py` CISPO preset.

Stages added after this brief, with their own briefs:

| Stage | Tasks | Brief |
|---|---|---|
| 5 — speculative decoding | 9 EAGLE-4 tree kernels (`eagle_*`) | `EAGLE4_kernels_task.md`, `EAGLE4_STAGE5.md` |
| 6 — multi-LoRA serving | 6 BGMV/SGMV kernels (`lora_*`) | `MULTI_LORA_kernels_task.md`, `MULTI_LORA_STAGE6.md` |
| 7 — TileGym ports | `relu_squared`, `silu_and_mul`, `geglu_tanh`, `rotary_embedding` | `TILEGYM_kernels_task.md` |

The three slowest Stage 5 Triton solutions were rewritten as tuned tensor-core kernels on 2026-11-14 (README, "Stage 5: tuned Triton solutions"); their CUDA C++ versions are still untuned.

## Per-task deliverable checklist

For each task: `tasks/<name>/model.py` (reference + `get_inputs`), honest `model_new_<backend>.py` + `kernels/<backend>/<name>.py` for at least Triton and CUDA, reward-hack and wrong-output fixtures, a catalog entry in `utils/task_catalog.py` (cases, sources, backends, stretch backends), task README (source suite, license, shape/dtype grid, recorded baseline numbers with the command and run ID that produced them), and suite coverage through `scripts/validate --suite`. Update `VALIDATION.md`/`KNOWN_ISSUES.md` with measured results only — never invented numbers.

## Stop conditions

ASK before: changing verification tolerances, reward semantics or timing meanings; editing protected root files; adding dependencies; launching any paid Modal GPU run; or restarting a GPU job that was killed. If no GPU is available, stop after making everything runnable and emit one copy-paste Modal command block.

---

Two notes from the original draft, kept for context:

1. **Reordered priorities**: the earlier draft put FlashAttention at P1 and GEMM at P0. That order was kept, but the GEMM task was made explicitly Tensor Core (`tl.dot` + WMMA + CuTe MMA atoms), since "tensor core kernel" as a separate task is really just GEMM done properly.
2. **Licensing flag inline** for GPU MODE (eval-only use), and CUTLASS/NVBench made *methodology* sources rather than task sources — candidates can't call CUTLASS kernels anyway under the provenance rules, so its only valid role is baseline ceilings.
