# Refined Prompt: Extending AgenticRL-CUDA-kernels Beyond AXPBY

## Role

You are a GPU kernel engineer extending the `AgenticRL-CUDA-kernels/` evaluation environment (repo: `ElenaViewSynthesis/RL-Army-of-Agents`, branch `main`). The environment currently ships exactly one workload — element-wise `alpha*a + b` on `(1,128)` fp32 — across five backends (CUDA, Triton, TileLang, CuTe DSL, cuTile). Your job is to add a **progressive task suite** of new kernel workloads, each conforming to the existing task contract, so the RL candidate-evaluation pipeline has non-trivial kernels to optimize.

## Hard constraints (from the existing harness — do not violate)

- Each task is a workdir shaped like the root: a protected `model.py` (PyTorch reference + `get_inputs()`), `binding.cpp`, `binding_registry.h`, `kernels/`, and a candidate `model_new.py`. Root `model.py` stays untouched; new tasks go under `tasks/<name>/` per `Benchmark_diagnostics_task.md` Step 5.
- `ModelNew` must match the reference constructor and `state_dict`; `forward` must enqueue all device work on `torch.cuda.current_stream()`.
- No `torch.*` compute ops in `model_new.py` or wrappers — allocation and metadata only. Compute belongs in authored kernels (`@triton.jit`, `@cute.kernel`, `__global__`, `T.prim_func`, `@ct.kernel`).
- No prewritten library kernels: no `triton.ops`, flash-attn packages, cuBLAS/cuDNN/CUTLASS-library calls. `tl.dot` is allowed; CuTe DSL GEMMs must be built from MMA/copy atoms, not `cutlass.utils` templates.
- Verification: 5 random inputs, atol/rtol = 1e-2. **ASK before changing tolerances** for reduced-precision tasks.
- Provenance: ≥90% of non-memory device time must be attributed to authored kernel names. Keep default symbol naming.
- Every task needs three fixture variants: honest implementation, a Torch-op reward-hack variant (must pass verification, fail provenance), and ideally a wrong-output variant — mirroring `tests/support.py:make_task`/`make_spoof_task`.
- Results stay flat and JSON-safe; no new Python deps beyond `torch>=2.3`, `triton>=3.0` (TileLang/CuTe DSL/cuTile have their own requirements files).
- GPU work runs on Modal H200; the dev host has no GPU. Everything must degrade to explicit "unavailable" on CPU.

## Source benchmark suites and sample tasks to extract

For each suite below, port the *task definition* (operator contract, shape grid, dtype set, correctness rule) — not their harness code. Record provenance of each task (source suite + URL + license) in the task's README.

### 1. KernelBench (highest priority — task-format model)

License: MIT-style research. Extract 3 samples:

- **Level 1 — `fused_linear_scale` / `gemm_div_sum_scale`**: `C = (A @ B) / s` then column-sum-scale epilogue. This is the deferred Level-2 GEMM already named in your README — implement it first. Shapes: `M,N,K ∈ {256, 512, 1024}`, fp32 + bf16.
- **Level 2 — fused reduction chain**: `x → LayerNorm → residual add → GELU`. Teaches multi-op fusion within one kernel launch.
- **Level 3 — attention block**: full `softmax(QK^T/√d)V` as the reference `model.py` — becomes the FlashAttention task below.

### 2. GPU MODE reference kernels (highest priority — rigor model)

License: research-reciprocity — **review terms before using problems as training data**; safe to use as eval-only tasks. Extract 3 samples:

- **Trimmed mean / masked reduction**: row-wise mean over variable-length rows with a length tensor — directly relevant to RL sequence masking. Shapes: `B ∈ {64, 512}`, `L ∈ {128..8192}` ragged.
- **Sort/scan**: block-level prefix sum (Blelloch or warp-scan), `N ∈ {1e5..1e7}` — exercises shared memory and inter-warp sync.
- **INT8/FP8 quantized GEMM** (defer to Stage 4): contract only for now.

### 3. TritonBench (high priority — backend parity)

Extract 3 samples:

- **`tl.dot` GEMM with epilogue**: `C = A@B + bias`, autotuned over `BLOCK_M/N/K`, `num_warps`, `num_stages`. This is the **primary Tensor Core task** — `tl.dot` on fp16/bf16 inputs with fp32 accumulation lowers to MMA on H200. Shape grid: square `{512,1024,2048,4096}` plus skinny `M ∈ {8,64}` × `N=K=4096` (RL-inference-like).
- **RMSNorm forward**: row reduction + scale, `H ∈ {1024..8192}`, fp32 accumulate on bf16 input.
- **Grouped GEMM**: `G ∈ {4,8}` expert matrices, per-group `M` — MoE-relevant, Triton-only initially.

### 4. FlashInfer-Bench (high priority — realistic LLM workloads)

Extract 3 samples:

- **RMSNorm + residual fused** (prefill shape): `(B*S, H)` with `S` variable.
- **Top-k / top-p sampling kernel**: logits `(B, V=128256)` → sampled token — rollout-generation-relevant.
- **Decode attention (single-query)**: `H_q=32, H_kv=8` GQA, `S_kv ∈ {512..8192}` — the decode-side counterpart to FlashAttention.

### 5. CUTLASS profiler (baseline, not a task source)

Use `cutlass_profiler` output **only** to record reference TFLOP/s ceilings per GEMM shape/dtype in the task README — candidates still may not call CUTLASS-library kernels. This gives honest "percent of cuBLAS" reporting instead of speedup-vs-eager only.

### 6. NVBench (measurement methodology)

Don't port tasks; adopt its conventions in `utils/benchmark.py` extensions: parameter sweeps, cold vs. warm timing, achieved bandwidth/FLOP/s fields in `bench/*` diagnostics.

### 7. Official Triton tutorials (implementation references)

Use as the *honest fixture* starting points: `fused-softmax`, `matmul` (tl.dot), `layer-norm`, `fused-attention` (the FlashAttention tutorial), `grouped-gemm`, `persistent-matmul`. Rewrite into the repo's `kernels/triton/` style — tutorials are reference material, not copy-paste solutions.

## The two flagship tasks (specify these in full)

### Task A — `tasks/flash_attention_fwd/`

- **Reference `model.py`**: `O = softmax(Q@K^T / sqrt(d)) @ V`, causal flag on, `B=4, H=8, S ∈ {512,1024,2048}, d=64`, bf16 inputs, fp32 softmax accumulation. `get_inputs()` returns `(Q,K,V)`.
- **Honest fixtures**: Triton (online-softmax tiled, from the official tutorial pattern); CUDA (two-pass shared-memory tiled, acceptable to be slower); CuTe DSL (MMA atoms + online softmax — stretch); TileLang (`T.gemm` + pipelined softmax — stretch).
- **Verification**: atol/rtol 1e-2 on bf16 is tight — **ASK** whether to relax to 2e-2 for this task only.
- **Benchmark**: report TFLOP/s (`4·B·H·S²·d` for causal ≈ half) and % of a recorded cuBLAS/FA2 ceiling.
- **Reward-hack fixture**: `torch.nn.functional.scaled_dot_product_attention` in `model_new.py` — must pass verification, fail provenance (this is exactly the attack the gate exists for).

### Task B — `tasks/gemm_tensor_core/`

- **Reference `model.py`**: `C = A @ B + bias`, `A:(M,K) bf16, B:(K,N) bf16, C:(M,N) fp32`, grid `{512,1024,2048,4096}²` plus `M=64` skinny case.
- **Honest fixtures**: Triton `tl.dot` (fp32 `acc`, `allow_tf32` off for fp32 variant); CUDA WMMA/`mma.sync` m16n8k16 kernel (the explicit Tensor Core path — shared-memory double-buffered tiles); CuTe DSL via `make_tiled_mma` atoms.
- **Acceptance**: honest Triton fixture ≥60% of recorded cuBLAS TFLOP/s at 4096³ on H200; provenance = 1.0; hack fixture fails provenance.

## Curriculum ordering (implement in this sequence)

| Stage | Tasks | New capability exercised |
|---|---|---|
| 1 — fundamentals | `row_softmax`, `masked_mean` (GPU MODE), `prefix_sum` | reductions, masks, shared memory |
| 2 — training kernels | `gemm_tensor_core` (Task B), `rmsnorm`, `log_softmax_gather` (token log-probs), `fused_adamw` | `tl.dot`/MMA, multi-pass fusion |
| 3 — RL-specific | `grpo_group_norm` (per-group mean/std over reward groups), `cispo_clip_weight` (clipped IS weight × advantage, token-level), `packed_varlen_reduction` | the actual GRPO/DAPO/CISPO math as kernels — unique to this repo, no external source needed |
| 4 — advanced | `flash_attention_fwd` (Task A), `grouped_gemm`, `topk_sampling`, `fp8_gemm` (contract only) | online softmax, MMA pipelines, MoE |

Stage 3 is the differentiator: it connects kernel optimization to the RL objective itself and no external suite covers it — derive contracts from `utils/grpo.py` and `utils/config.py` CISPO preset.

## Per-task deliverable checklist

For each task: `tasks/<name>/model.py` (reference + `get_inputs`), honest `model_new_<backend>.py` + `kernels/<backend>/<name>.py` for at least Triton and CUDA, reward-hack fixture, task README (source suite, license, shape/dtype grid, recorded baseline numbers with the command that produced them), and an entry in `scripts/validate` suite mode. Update `VALIDATION.md`/`KNOWN_ISSUES.md` with measured results only — never invented numbers.

## Stop conditions

ASK before: changing verification tolerances, editing protected root files, adding dependencies, or if no GPU is available (then stop after making everything runnable and emit one copy-paste Modal command block).

---

Two notes on your draft that I changed deliberately:

1. **Reordered priorities**: your table put FlashAttention at P1 and GEMM at P0 — I kept that, but made the GEMM task explicitly Tensor Core (`tl.dot` + WMMA + CuTe MMA atoms), since "tensor core kernel" as a separate task is really just GEMM done properly.
2. **Added the licensing flag inline** for GPU MODE (eval-only use) and made CUTLASS/NVBench *methodology* sources rather than task sources — candidates can't call CUTLASS kernels anyway under your provenance rules, so its only valid role is baseline ceilings.
