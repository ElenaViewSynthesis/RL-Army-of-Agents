# Task: Stage 6 multi-LoRA kernels (BGMV + SGMV shrink/expand) as engine-free RL tasks

You are working in AgenticRL-CUDA-kernels/ (WSL2). Read AGENTS.md, the binding
Ground rules in Benchmark_diagnostics_task.md, README.md "Kernel tasks the
agent works on", EAGLE4_STAGE5.md, and an existing Stage 5 task such as
tasks/eagle_tree_step/ before writing anything. Match the Stage 5 task
format exactly: a trusted model.py whose inputs come from the catalog, honest
Triton and CUDA solutions, negative controls, and catalog contracts in
Torch-free modules.

The goal is the same as for EAGLE-4: turn the core multi-LoRA serving
kernels into standalone workloads the RL environment can score, with no
serving framework involved. The original serving-integration brief is kept
in Appendix A for reference only. Framework wrappers, tensor-parallel
communication, custom-op registration and end-to-end serving benchmarks are
**not** part of Stage 6.

## Operation (unchanged from the original brief)

For token t with adapter slot i = idx[t] (i = -1 means no LoRA) and output
slice s (1 slice for q/o/down, 2 for gate_up, 3 for qkv):

```
shrink:  buf[s, t, :r_i] = scale_i * x[t] @ A[i, s, :r_i, :].T      # fp32 output
expand:  y[t, off_s : off_s + n_s] += buf[s, t, :r_i] @ B[i, s, :, :r_i].T
```

Preallocated, static shapes:
- x: [num_tokens, hidden_in] fp16/bf16
- A: [max_loras, num_slices, max_rank, hidden_in]
- B: [max_loras, num_slices, max_out_slice, max_rank]
- idx: [num_tokens] int32; ranks: [max_loras] int32; scales: [max_loras] fp32
- buf: [num_slices, num_tokens, max_rank] fp32
- max_rank is padded to a multiple of 16.

## Stage 6 tasks (one tasks/<name>/ folder each, stage 6 in the catalog)

1. **lora_shrink_bgmv:** decode shrink, one token per sequence. Memory-bound
   with N = rank, so split-K over hidden_in is expected. Output: fp32 buf.
2. **lora_expand_bgmv:** decode expand, tiled over the output dim. Output:
   y_in + delta, downcast only on the add into y.
3. **lora_segment_metadata:** given idx, build the sorted token order and
   per-segment adapter id, start and length, plus the active-segment count,
   on the device with no host sync. All outputs are integers and compared
   exactly. Choose a stable order and document it in the contract.
4. **lora_shrink_sgmv:** prefill/mixed shrink over adapter segments (inputs
   include the metadata from task 3, built by the trusted reference). Each
   segment is a small GEMM that reuses A tiles across its tokens.
5. **lora_expand_sgmv:** prefill/mixed expand over segments; reuses B tiles.
6. **lora_fused_sgmv** (stretch): shrink then expand in one task with the
   fp32 intermediate kept on chip or in fp32 scratch. It must stay
   numerically equivalent to tasks 4+5.

The slice count (1/2/3) and per-slice output offsets are case parameters
of every shrink/expand task, not separate tasks.

## Correctness rules for every task

- Accumulate in fp32 and keep buf in fp32 end to end; downcast only when
  adding into y.
- Variable rank per slot, 8 to 256 (512 as a stretch case), including
  **mixed ranks in one batch**. Use `ranks` to skip padded work.
- **Poisoned padding:** the trusted inputs fill the padded rank region of A
  and B (columns r_i..max_rank) with NaN, so any read of padding poisons the
  output and fails verification. Do not change this.
- **idx = -1 tokens** get zero work and zero writes. The trusted inputs give
  those rows of y_in large sentinel values and make the LoRA delta large, so
  any write to them fails the standard tolerance.
- Adapters with a dtype different from the model's are cast once, by the
  trusted input builder, never inside the kernel.
- No host syncs (`.item()`, `.tolist()`, `.nonzero()`) and no data-dependent
  shapes in any honest `forward`. The suite already checks CUDA-graph
  capture (`--benchmark --cuda-graphs`). Honest solutions must be
  graph-capturable, and replay must be correct after the slot contents
  change.

## Lessons from Stage 5 (binding)

- **Trusted references must be vectorized GPU code.** Gather adapter weights
  with `A[idx]`-style indexing and masks; no per-token Python loop with host
  syncs. A per-token fp32 loop may exist only as a test oracle (as
  `tests/fixtures/eagle_reference_loop.py` does for Stage 5), with a test
  proving the reference matches it exactly. Stage 5's loop references made
  the eager baseline meaningless and overflowed the profiler record at large
  cases.
- **Use 64-bit offsets** wherever num_tokens × hidden × slices × rank or a
  split count can exceed 2^31 (Stage 5 verify attention crashed with an
  illegal memory access from int32 offsets).
- **Grid axes 1 and 2 are capped at 65535.** Put large dimensions (tokens,
  hidden tiles) on axis 0.
- The profiler now writes per-event samples to a sidecar file, but keep the
  reference's kernel count reasonable anyway: it is the eager baseline.

## Case grid (catalog cases; smoke first)

- num_tokens {1, 3, 64, 512, 4096}; hidden_in/out {2048, 4096, 5120, 8192,
  14336}; ranks {8, 16, 32, 64, 128, 256} plus mixed-rank batches;
  max_loras {1, 4, 16}; slices {1, 2, 3}; dtype {fp16, bf16}.
- idx patterns: no LoRA tokens, some -1, all -1, all tokens on one adapter,
  all tokens on distinct adapters.
- BGMV tasks use the decode sizes (num_tokens 1–256); SGMV tasks use the
  prefill sizes (512–8192).
- Do not take the full product. Define the cases as Stage 5 did, with
  `smoke` = the smallest case and `acceptance` = the largest prefill case
  (4096 tokens, 14336 hidden, rank 256, 16 slots, 3 slices). Add a
  deterministic `stratified` selection that covers every axis value.

## Fixtures (every task)

- honest Triton; honest CUDA where practical. Triton must run on sm_75+ with
  a small fixed set of configs chosen by shape heuristics, not broad
  autotuning.
- `hack`: a Torch-op substitution (expect `provenance`).
- `wrong`: a wrong output (expect `verification`).
- `wrong_padding`: reads the padded rank region (expect `verification`,
  through the NaN poisoning).
- `wrong_skip`: writes to idx = -1 rows (expect `verification`, through the
  sentinels).
- `fp16_intermediate` (diagnostic): keeps buf in fp16 at rank ≥ 64. Report
  its error distribution next to the honest one. **ASK** before turning it
  into an expected-failure gate, since whether it exceeds the standard
  tolerance depends on the case.

## Tolerances and gates

- Use the existing verification: atol/rtol 1e-2, with integer outputs
  exact (`exact_integer_outputs`, approved). The original brief's relative
  bounds (bf16 ≤ 1e-2·max|ref|, fp16 ≤ 5e-3·max|ref|) are a different
  metric. Record them as diagnostics in the case JSON, and **ASK** before
  making them a gate or changing tolerances.
- The authored-only provenance gate applies: no cuBLAS/cuDNN time in honest
  solutions.

## Diagnostics

- `workload()` FLOP/byte conventions: shrink logical bytes (x + touched A
  rows + buf) for GB/s against the HBM roofline; expand FLOPs (2·tokens·
  rank·n_out per slice) for TFLOP/s. Masked padding is excluded from both.
- Report compile/JIT time cold and warm (`--cold-start`) and graph
  capturability (`--cuda-graphs`).

## Sources and licensing

Write from the published behaviour only: punica (BGMV/SGMV, Apache-2.0),
S-LoRA (paper and Apache-2.0 code) and vLLM's LoRA kernels (Apache-2.0). Do
not copy their code. Add a `SOURCES` entry per origin with its license. These
are permissive licenses, so the tasks need not be evaluation-only unless
your license check finds otherwise. Report the finding.

## Verification order

1. Contracts, references and tests (Torch-free where possible), with the
   loop oracle proving each reference.
2. Both local suites green.
3. **ASK** before any Modal H200 run. Then smoke, then acceptance, with the
   frozen-source flow and full evidence downloads into `artifacts/runs/`.
   CPU-interpreter results are not acceptable GPU evidence.

Do not commit: Claude commits from the Windows checkout. Update README.md
(the stage and per-task tables), VALIDATION.md and KNOWN_ISSUES.md.

## Report back

Files added, the license finding, raw test summaries, per-task results, what
is unverified and why, and open decisions (tolerance metric, fp16
intermediate gate, any interface question).

---

# Appendix A: original serving-integration brief (not part of Stage 6)

Kept unchanged for reference. Framework integration, tensor-parallel
sharding, custom-op registration and end-to-end serving benchmarks are out
of scope for the RL task suite.

### Role
You are a GPU kernel engineer. Write production-quality kernels, PyTorch bindings, tests and
benchmarks for serving many LoRA adapters concurrently on one base model, in the style of
punica's BGMV/SGMV. Target framework: <vLLM | SGLang> at commit <SHA>. Read its existing LoRA
layer wrappers (column/row/QKV/merged-column parallel) before writing code, and match their
interfaces.

### Background (what exists and why it's insufficient)
The current implementation uses vendored punica BGMV CUDA kernels. These kernels:
- are inefficient for prefill (they run one token per program, with no reuse of A/B across tokens of the same adapter)
- require sm_80+
- cap the LoRA rank at 64
- have long compile times
- do not shard LoRA compute under tensor parallelism

The CUTLASS SGMV alternative was rejected for two reasons: it isn't configurable enough, and it
stores the intermediate (x @ A) in fp16, which loses accuracy.

### Operation definition
For token t with adapter index i = idx[t] (i = -1 means no LoRA) and output slice s
(1 slice for q/o/down, 2 for gate_up, 3 for qkv):

```
shrink:  buf[s, t, :r_i]  = scale_i * x[t] @ A[i, s, :r_i, :].T      # fp32 output
expand:  y[t, off_s : off_s + n_s] += buf[s, t, :r_i] @ B[i, s, :, :r_i].T
```

Shapes (preallocated slots, static across steps):
- x:   [num_tokens, hidden_in]                     fp16/bf16
- A:   [max_loras, num_slices, max_rank, hidden_in]
- B:   [max_loras, num_slices, max_out_slice, max_rank]
- idx: [num_tokens] int32; ranks: [max_loras] int32; scales: [max_loras] fp32
- buf: [num_slices, num_tokens, max_rank] fp32

### Kernel requirements
1. Two kernel families:
   - BGMV for decode (one token per sequence). Shrink is memory-bound with a tiny N (= rank), so use split-K over hidden_in. Expand tiles over the output dim.
   - SGMV for prefill/mixed batches. Group tokens into segments by adapter, then tile each segment as a small GEMM so that A/B tiles are reused across all tokens of the same adapter.
2. Accumulate in fp32 and keep the intermediate buffer in fp32 end to end. Downcast only when adding into y.
3. Variable rank per adapter, from 8 up to at least 256 (ideally 512). Pad max_rank to a multiple of 16. Use the per-slot `ranks` tensor to mask or skip the padded work, and never read garbage from padding.
4. Handle idx = -1 tokens with zero work and zero writes.
5. Write the kernels in Triton so they run on sm_75+ (T4, A10, A100, H100). Choose a small, fixed set of configs via shape heuristics rather than broad autotuning, which keeps JIT/compile time low. A hand-written CUDA/CuTe path for sm_90 is optional and comes only after Triton is correct.
6. Inputs come from the activations only. The kernels must not depend on the base weight format, so they must work unchanged with a quantized base (AWQ, GPTQ, FP8).
7. Adapters with a different dtype from the model are cast once, at load time into the slot, never in the kernel.

### torch.compile / CUDA graph contract
- No host syncs (`.item()`, `.tolist()`, `.nonzero()`) and no data-dependent shapes inside the forward pass.
- Compute the segment metadata (sorted token order, per-segment adapter id, start, length, number of active segments) once per step, before the forward pass. Write it into persistent, max-size device buffers. Kernels read the active-segment count from a device tensor and early-exit on unused grid programs.
- Register each op with `torch.library.custom_op`, include a `register_fake` implementation, and mark in-place mutation of `y`.
- The ops must capture cleanly into a CUDA graph at every captured batch size, and replay must be correct when the active adapters change between steps.

### Tensor parallelism (S-LoRA-style sharding)
Implement a `fully_sharded` mode alongside the default replicated mode:
- Column-parallel layers (qkv, gate_up): shard A along rank → shrink → all-gather buf (tiny: tokens × rank) → expand with B sharded along output. No other communication.
- Row-parallel layers (o, down): x arrives sharded along hidden_in → shrink gives partial sums → all-reduce buf (tiny) → expand with B sharded along output, adding only into this rank's output shard *before* the base layer's all-reduce, so the existing all-reduce sums correctly.
- Never add the full LoRA delta on every TP rank before an all-reduce. That double-counts it.

### Testing (pytest)
- Reference implementation: a per-token PyTorch loop in fp32. Also run an end-to-end check that merges a single LoRA into the base weights and compares the logits of the merged model against the LoRA path (greedy tokens should match over 64 steps).
- Test matrix: num_tokens {1, 3, 64, 512, 4096}; hidden_in/out {2048, 4096, 5120, 8192, 14336}; ranks {8, 16, 32, 64, 128, 256}, including *mixed ranks in one batch*; max_loras {1, 4, 16}; slices {1, 2, 3}; dtype {fp16, bf16}; some or all tokens with idx = -1; batches where all tokens share one adapter vs. all distinct adapters; TP {1, 2, 4}, sharded vs. replicated.
- Tolerance: bf16 max-abs-err ≤ 1e-2 · max|ref|, fp16 ≤ 5e-3 · max|ref|. Report the error distribution, not just pass/fail.
- Regression test that the fp32 intermediate fixes the half-precision accuracy drop: compare against a deliberately fp16-intermediate variant at rank 64+.
- CUDA-graph tests: capture, swap adapters in the slots, replay, and check correctness. Also run `torch.compile(fullgraph=True)` over a toy LoRA layer.

### Benchmarks
- Kernel level: decode (batch 1–256) and prefill (512–8192 tokens). Report µs/call, achieved HBM GB/s for shrink against the roofline, and TFLOPs for SGMV expand. Compare against the vendored punica BGMV and the PyTorch reference.
- End-to-end: throughput and TTFT/ITL at 1 vs. 8 vs. 32 adapters, with the overhead stated relative to the base model with no LoRA.
- Report compile/JIT time, cold and warm.

### Deliverables
1. The kernels (`lora_shrink`, `lora_expand`, BGMV and SGMV variants) plus the metadata builder
2. Custom-op registration and integration into the framework's LoRA layer wrappers
3. The test suite and a benchmark script that writes a markdown table
4. A short design note covering tiling choices per regime, the split-K heuristic, the TP communication volume, and known limitations

### Non-goals (explicitly out of scope)
- Unified paging of LoRA weights with the KV cache. Keep the fixed preallocated slots.
- Auto-downloading adapters from S3/HF, and overlapping load/unload with the forward pass.
- Changing the scheduler policy.

### Working style
Get correctness first with the simplest Triton kernel, then optimize one regime at a time and
re-run the full test suite after each change. If a requirement conflicts with the framework's
existing interfaces, stop and report the conflict rather than silently changing the interface.
