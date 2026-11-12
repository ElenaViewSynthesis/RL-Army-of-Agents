# `lora_expand_sgmv`

## Operator contract

FP32 accumulated multi-LoRA; NaN rank padding masked; skipped rows preserved; disjoint output slices



The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- punica: https://github.com/punica-ai/punica; license: [Apache-2.0](https://github.com/punica-ai/punica/blob/591b59899f0a20760821785d06b331c8a2e5cb86/LICENSE).
- slora: https://github.com/S-LoRA/S-LoRA; license: [Apache-2.0](https://github.com/S-LoRA/S-LoRA/blob/c1ddf488781ea7f551cd0bb07bfd097124c93411/LICENSE).
- vllm: https://github.com/vllm-project/vllm; license: [Apache-2.0](https://github.com/vllm-project/vllm/blob/4e3463c87493ebd2c5d5fbe4be6469911e23e2bf/LICENSE).

Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
| `tokens512_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_float16` | `float16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 8, 'slots': 1, 'slices': 1, 'pattern': 'single', 'mixed': False}` |
| `tokens512_hidden2048_out2048_rank8_slots1_slices1_patternsingle_mixedfalse_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 8, 'slots': 1, 'slices': 1, 'pattern': 'single', 'mixed': False}` |
| `tokens512_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_float16` | `float16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 8, 'slots': 4, 'slices': 1, 'pattern': 'mixed', 'mixed': False}` |
| `tokens512_hidden2048_out2048_rank8_slots4_slices1_patternmixed_mixedfalse_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 8, 'slots': 4, 'slices': 1, 'pattern': 'mixed', 'mixed': False}` |
| `tokens512_hidden4096_out4096_rank16_slots1_slices2_patternsingle_mixedfalse_float16` | `float16` | `{'tokens': 512, 'hidden': 4096, 'out': 4096, 'rank': 16, 'slots': 1, 'slices': 2, 'pattern': 'single', 'mixed': False}` |
| `tokens512_hidden4096_out4096_rank16_slots1_slices2_patternsingle_mixedfalse_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 4096, 'out': 4096, 'rank': 16, 'slots': 1, 'slices': 2, 'pattern': 'single', 'mixed': False}` |
| `tokens512_hidden5120_out5120_rank32_slots16_slices3_patternround_robin_mixedtrue_float16` | `float16` | `{'tokens': 512, 'hidden': 5120, 'out': 5120, 'rank': 32, 'slots': 16, 'slices': 3, 'pattern': 'round_robin', 'mixed': True}` |
| `tokens512_hidden5120_out5120_rank32_slots16_slices3_patternround_robin_mixedtrue_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 5120, 'out': 5120, 'rank': 32, 'slots': 16, 'slices': 3, 'pattern': 'round_robin', 'mixed': True}` |
| `tokens512_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_float16` | `float16` | `{'tokens': 512, 'hidden': 8192, 'out': 8192, 'rank': 64, 'slots': 4, 'slices': 2, 'pattern': 'distinct', 'mixed': False}` |
| `tokens512_hidden8192_out8192_rank64_slots4_slices2_patterndistinct_mixedfalse_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 8192, 'out': 8192, 'rank': 64, 'slots': 4, 'slices': 2, 'pattern': 'distinct', 'mixed': False}` |
| `tokens512_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_float16` | `float16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 128, 'slots': 16, 'slices': 1, 'pattern': 'none', 'mixed': True}` |
| `tokens512_hidden2048_out2048_rank128_slots16_slices1_patternnone_mixedtrue_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 128, 'slots': 16, 'slices': 1, 'pattern': 'none', 'mixed': True}` |
| `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | `float16` | `{'tokens': 512, 'hidden': 4096, 'out': 4096, 'rank': 256, 'slots': 16, 'slices': 3, 'pattern': 'mixed', 'mixed': True}` |
| `tokens512_hidden4096_out4096_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 4096, 'out': 4096, 'rank': 256, 'slots': 16, 'slices': 3, 'pattern': 'mixed', 'mixed': True}` |
| `tokens512_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | `float16` | `{'tokens': 512, 'hidden': 14336, 'out': 14336, 'rank': 256, 'slots': 16, 'slices': 3, 'pattern': 'mixed', 'mixed': True}` |
| `tokens512_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 14336, 'out': 14336, 'rank': 256, 'slots': 16, 'slices': 3, 'pattern': 'mixed', 'mixed': True}` |
| `tokens512_hidden2048_out2048_rank64_slots4_slices1_patternsingle_mixedfalse_float16` | `float16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 64, 'slots': 4, 'slices': 1, 'pattern': 'single', 'mixed': False}` |
| `tokens512_hidden2048_out2048_rank64_slots4_slices1_patternsingle_mixedfalse_bfloat16` | `bfloat16` | `{'tokens': 512, 'hidden': 2048, 'out': 2048, 'rank': 64, 'slots': 4, 'slices': 1, 'pattern': 'single', 'mixed': False}` |
| `tokens4096_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_float16` | `float16` | `{'tokens': 4096, 'hidden': 14336, 'out': 14336, 'rank': 256, 'slots': 16, 'slices': 3, 'pattern': 'mixed', 'mixed': True}` |
| `tokens4096_hidden14336_out14336_rank256_slots16_slices3_patternmixed_mixedtrue_bfloat16` | `bfloat16` | `{'tokens': 4096, 'hidden': 14336, 'out': 14336, 'rank': 256, 'slots': 16, 'slices': 3, 'pattern': 'mixed', 'mixed': True}` |

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
python -m scripts.validate --suite lora_expand_sgmv --device cuda --purpose eval \
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

See [Stage 6 contracts](../../MULTI_LORA_STAGE6.md) for exact layouts, controls and limits.
