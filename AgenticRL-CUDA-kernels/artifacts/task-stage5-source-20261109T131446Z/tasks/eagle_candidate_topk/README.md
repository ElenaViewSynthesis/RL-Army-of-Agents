# `eagle_candidate_topk`

## Operator contract

Vocabulary shard top-k with lower-ID ties, then selected-row fp32 gather-dot re-scoring

See [Stage 5 contracts](../../EAGLE4_STAGE5.md) for the exact layouts, math, tie rules, history conventions, workload counts and source deviations. CUDA and Triton sources are provided; GPU acceptance is not yet measured. Use `--fixtures all` for the wrong-ancestor control and, for GDN on Triton, chain replay.

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- EAGLE-4 documented kernel behavior (evaluation-only adaptation): https://github.com/eagle-hq/EAGLE-4/blob/main/docs/kernels.md; license: [Eagle 4 Source-Available License](https://github.com/eagle-hq/EAGLE-4/blob/main/LICENSE).

**Evaluation only; training clearance not recorded.** This project includes or is derived from Eagle 4 technology developed by Eagle AI Corp. It is not an official Eagle AI Corp project. The Eagle 4 Source-Available License is not OSI-approved. See [LICENSE](../../legal/eagle4/LICENSE), [NOTICE](../../legal/eagle4/NOTICE), [Trademark Policy](../../legal/eagle4/TRADEMARKS.md) and [conditions/acceptance notice](../../EAGLE4_STAGE5.md#source-license-and-attribution).

Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
| `b1_h4096_v32000_top_k8_bfloat16` | `bfloat16` | `{'b': 1, 'h': 4096, 'v': 32000, 'top_k': 8}` |
| `b1_h4096_v32000_top_k16_bfloat16` | `bfloat16` | `{'b': 1, 'h': 4096, 'v': 32000, 'top_k': 16}` |
| `b1_h4096_v128256_top_k8_bfloat16` | `bfloat16` | `{'b': 1, 'h': 4096, 'v': 128256, 'top_k': 8}` |
| `b1_h4096_v128256_top_k16_bfloat16` | `bfloat16` | `{'b': 1, 'h': 4096, 'v': 128256, 'top_k': 16}` |
| `b1_h4096_v151936_top_k8_bfloat16` | `bfloat16` | `{'b': 1, 'h': 4096, 'v': 151936, 'top_k': 8}` |
| `b1_h4096_v151936_top_k16_bfloat16` | `bfloat16` | `{'b': 1, 'h': 4096, 'v': 151936, 'top_k': 16}` |
| `b8_h4096_v32000_top_k8_bfloat16` | `bfloat16` | `{'b': 8, 'h': 4096, 'v': 32000, 'top_k': 8}` |
| `b8_h4096_v32000_top_k16_bfloat16` | `bfloat16` | `{'b': 8, 'h': 4096, 'v': 32000, 'top_k': 16}` |
| `b8_h4096_v128256_top_k8_bfloat16` | `bfloat16` | `{'b': 8, 'h': 4096, 'v': 128256, 'top_k': 8}` |
| `b8_h4096_v128256_top_k16_bfloat16` | `bfloat16` | `{'b': 8, 'h': 4096, 'v': 128256, 'top_k': 16}` |
| `b8_h4096_v151936_top_k8_bfloat16` | `bfloat16` | `{'b': 8, 'h': 4096, 'v': 151936, 'top_k': 8}` |
| `b8_h4096_v151936_top_k16_bfloat16` | `bfloat16` | `{'b': 8, 'h': 4096, 'v': 151936, 'top_k': 16}` |
| `b32_h4096_v32000_top_k8_bfloat16` | `bfloat16` | `{'b': 32, 'h': 4096, 'v': 32000, 'top_k': 8}` |
| `b32_h4096_v32000_top_k16_bfloat16` | `bfloat16` | `{'b': 32, 'h': 4096, 'v': 32000, 'top_k': 16}` |
| `b32_h4096_v128256_top_k8_bfloat16` | `bfloat16` | `{'b': 32, 'h': 4096, 'v': 128256, 'top_k': 8}` |
| `b32_h4096_v128256_top_k16_bfloat16` | `bfloat16` | `{'b': 32, 'h': 4096, 'v': 128256, 'top_k': 16}` |
| `b32_h4096_v151936_top_k8_bfloat16` | `bfloat16` | `{'b': 32, 'h': 4096, 'v': 151936, 'top_k': 8}` |
| `b32_h4096_v151936_top_k16_bfloat16` | `bfloat16` | `{'b': 32, 'h': 4096, 'v': 151936, 'top_k': 16}` |

## Fixtures and gates

- Honest backends: triton, cuda.
- Stretch backends: none declared.
- `model_new.py` selects the default honest fixture. `model_new_hack.py` is an
  intentional Torch reference-compute attack; it must verify then fail provenance.
- `model_new_wrong.py` intentionally returns NaNs or invalid integer sentinels;
  it must fail verification, not merely compilation.
- Verification uses five fresh random inputs; floating outputs keep atol=rtol=1e-2.
  The progressive suite requires exact equality for integer outputs, including
  token IDs and prefix sums. `utils.task_verification` exposes the same strict
  check directly; stock `utils.verification` and AXPBY defaults stay unchanged.
- The suite alone applies authored-only provenance: at least 90% of non-memory
  device time, no library-kernel time, and 100% authored for gemm_tensor_core.
  Stock AXPBY attribution/reward rules are unchanged.
- No triton.ops, flash-attn packages, cuBLAS/cuDNN/CUTLASS-library candidate calls.
  tl.dot and authored CuTe MMA/copy atoms are allowed.
- Name matching remains a heuristic, not proof of authorship or a security sandbox.
- CPU/missing capabilities are **unavailable**, never GPU passes.

## Reproduction and measured baselines

From the project root, on an H200 runtime:

```bash
python -m scripts.validate --suite eagle_candidate_topk --device cuda --purpose eval \
  --case-set smoke --benchmark --cold-start --timeout-s 600
```

Use `--case-set all` for the declared grid, or an exact case ID. Output is in a
new artifacts/task-suite-* directory; progress.jsonl retains completed cells.
`--case-set acceptance` selects 4096^3 bf16 GEMM or S=2048 attention where applicable.
Warm event/wall samples and the separately defined first-forward sample are
diagnostics, never reward timings. The first-forward sample is not cache-flushed
and may include lazy compilation. Logical bandwidth is not measured HBM traffic.
cuBLAS, CUTLASS-profiler and Flash-provider baselines must retain their identities;
no CUTLASS result is relabeled cuBLAS and no unverified Flash provider is called FA2.

Unavailable: no measured GPU baseline has been recorded here.

No absent timing, TFLOP/s, bandwidth, ceiling percentage, or H200 acceptance is zero-filled.
