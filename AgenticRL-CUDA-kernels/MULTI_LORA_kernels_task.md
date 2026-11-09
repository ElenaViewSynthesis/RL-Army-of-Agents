# Task: Multi-LoRA GEMM kernels (shrink/expand, BGMV + SGMV) for batched LoRA serving

## Role
You are a GPU kernel engineer. Write production-quality kernels, PyTorch bindings, tests and
benchmarks for serving many LoRA adapters concurrently on one base model, in the style of
punica's BGMV/SGMV. Target framework: <vLLM | SGLang> at commit <SHA>. Read its existing LoRA
layer wrappers (column/row/QKV/merged-column parallel) before writing code, and match their
interfaces.

## Background (what exists and why it's insufficient)
The current implementation uses vendored punica BGMV CUDA kernels. These kernels:
- are inefficient for prefill (they run one token per program, with no reuse of A/B across tokens of the same adapter)
- require sm_80+
- cap the LoRA rank at 64
- have long compile times
- do not shard LoRA compute under tensor parallelism

The CUTLASS SGMV alternative was rejected for two reasons: it isn't configurable enough, and it
stores the intermediate (x @ A) in fp16, which loses accuracy.

## Operation definition
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

## Kernel requirements
1. Two kernel families:
   - BGMV for decode (one token per sequence). Shrink is memory-bound with a tiny N (= rank), so use split-K over hidden_in. Expand tiles over the output dim.
   - SGMV for prefill/mixed batches. Group tokens into segments by adapter, then tile each segment as a small GEMM so that A/B tiles are reused across all tokens of the same adapter.
2. Accumulate in fp32 and keep the intermediate buffer in fp32 end to end. Downcast only when adding into y.
3. Variable rank per adapter, from 8 up to at least 256 (ideally 512). Pad max_rank to a multiple of 16. Use the per-slot `ranks` tensor to mask or skip the padded work, and never read garbage from padding.
4. Handle idx = -1 tokens with zero work and zero writes.
5. Write the kernels in Triton so they run on sm_75+ (T4, A10, A100, H100). Choose a small, fixed set of configs via shape heuristics rather than broad autotuning, which keeps JIT/compile time low. A hand-written CUDA/CuTe path for sm_90 is optional and comes only after Triton is correct.
6. Inputs come from the activations only. The kernels must not depend on the base weight format, so they must work unchanged with a quantized base (AWQ, GPTQ, FP8).
7. Adapters with a different dtype from the model are cast once, at load time into the slot, never in the kernel.

## torch.compile / CUDA graph contract
- No host syncs (`.item()`, `.tolist()`, `.nonzero()`) and no data-dependent shapes inside the forward pass.
- Compute the segment metadata (sorted token order, per-segment adapter id, start, length, number of active segments) once per step, before the forward pass. Write it into persistent, max-size device buffers. Kernels read the active-segment count from a device tensor and early-exit on unused grid programs.
- Register each op with `torch.library.custom_op`, include a `register_fake` implementation, and mark in-place mutation of `y`.
- The ops must capture cleanly into a CUDA graph at every captured batch size, and replay must be correct when the active adapters change between steps.

## Tensor parallelism (S-LoRA-style sharding)
Implement a `fully_sharded` mode alongside the default replicated mode:
- Column-parallel layers (qkv, gate_up): shard A along rank → shrink → all-gather buf (tiny: tokens × rank) → expand with B sharded along output. No other communication.
- Row-parallel layers (o, down): x arrives sharded along hidden_in → shrink gives partial sums → all-reduce buf (tiny) → expand with B sharded along output, adding only into this rank's output shard *before* the base layer's all-reduce, so the existing all-reduce sums correctly.
- Never add the full LoRA delta on every TP rank before an all-reduce. That double-counts it.

## Testing (pytest)
- Reference implementation: a per-token PyTorch loop in fp32. Also run an end-to-end check that merges a single LoRA into the base weights and compares the logits of the merged model against the LoRA path (greedy tokens should match over 64 steps).
- Test matrix: num_tokens {1, 3, 64, 512, 4096}; hidden_in/out {2048, 4096, 5120, 8192, 14336}; ranks {8, 16, 32, 64, 128, 256}, including *mixed ranks in one batch*; max_loras {1, 4, 16}; slices {1, 2, 3}; dtype {fp16, bf16}; some or all tokens with idx = -1; batches where all tokens share one adapter vs. all distinct adapters; TP {1, 2, 4}, sharded vs. replicated.
- Tolerance: bf16 max-abs-err ≤ 1e-2 · max|ref|, fp16 ≤ 5e-3 · max|ref|. Report the error distribution, not just pass/fail.
- Regression test that the fp32 intermediate fixes the half-precision accuracy drop: compare against a deliberately fp16-intermediate variant at rank 64+.
- CUDA-graph tests: capture, swap adapters in the slots, replay, and check correctness. Also run `torch.compile(fullgraph=True)` over a toy LoRA layer.

## Benchmarks
- Kernel level: decode (batch 1–256) and prefill (512–8192 tokens). Report µs/call, achieved HBM GB/s for shrink against the roofline, and TFLOPs for SGMV expand. Compare against the vendored punica BGMV and the PyTorch reference.
- End-to-end: throughput and TTFT/ITL at 1 vs. 8 vs. 32 adapters, with the overhead stated relative to the base model with no LoRA.
- Report compile/JIT time, cold and warm.

## Deliverables
1. The kernels (`lora_shrink`, `lora_expand`, BGMV and SGMV variants) plus the metadata builder
2. Custom-op registration and integration into the framework's LoRA layer wrappers
3. The test suite and a benchmark script that writes a markdown table
4. A short design note covering tiling choices per regime, the split-K heuristic, the TP communication volume, and known limitations

## Non-goals (explicitly out of scope)
- Unified paging of LoRA weights with the KV cache. Keep the fixed preallocated slots.
- Auto-downloading adapters from S3/HF, and overlapping load/unload with the forward pass.
- Changing the scheduler policy.

## Working style
Get correctness first with the simplest Triton kernel, then optimize one regime at a time and
re-run the full test suite after each change. If a requirement conflicts with the framework's
existing interfaces, stop and report the conflict rather than silently changing the interface.
