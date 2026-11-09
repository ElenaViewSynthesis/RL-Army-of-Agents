# `layernorm_residual_gelu`

## Operator contract

Affine population LayerNorm, residual add, exact erf GELU; fp32 arithmetic then output cast

All normalization/affine/residual/GELU arithmetic is fp32 before the final output cast. Variance is population variance of centered values. GELU uses erf, not the tanh approximation. This is an adapted fused-operator workload, not an upstream harness port.

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- KernelBench: https://github.com/ScalingIntelligence/KernelBench; license: [MIT](https://github.com/ScalingIntelligence/KernelBench/blob/main/LICENSE).
- Official Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/index.html; license: [MIT](https://github.com/triton-lang/triton/blob/main/LICENSE).

Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
| `b64_h1024_float32` | `float32` | `{'b': 64, 'h': 1024}` |
| `b64_h4096_float32` | `float32` | `{'b': 64, 'h': 4096}` |
| `b64_h8192_float32` | `float32` | `{'b': 64, 'h': 8192}` |
| `b512_h1024_float32` | `float32` | `{'b': 512, 'h': 1024}` |
| `b512_h4096_float32` | `float32` | `{'b': 512, 'h': 4096}` |
| `b512_h8192_float32` | `float32` | `{'b': 512, 'h': 8192}` |
| `b64_h1024_bfloat16` | `bfloat16` | `{'b': 64, 'h': 1024}` |
| `b64_h4096_bfloat16` | `bfloat16` | `{'b': 64, 'h': 4096}` |
| `b64_h8192_bfloat16` | `bfloat16` | `{'b': 64, 'h': 8192}` |
| `b512_h1024_bfloat16` | `bfloat16` | `{'b': 512, 'h': 1024}` |
| `b512_h4096_bfloat16` | `bfloat16` | `{'b': 512, 'h': 4096}` |
| `b512_h8192_bfloat16` | `bfloat16` | `{'b': 512, 'h': 8192}` |

## Fixtures and gates

- Honest backends: triton, cuda.
- Stretch backends: none declared.
- `model_new.py` selects the default honest fixture. `model_new_hack.py` is an
  intentional Torch reference-compute attack; it must verify then fail provenance.
- `model_new_wrong.py` intentionally returns NaNs or invalid integer sentinels;
  it must fail verification, not merely compilation.
- Verification uses five fresh random inputs; floating outputs keep atol=rtol=1e-2.
  The progressive suite requires exact equality for integer outputs. The direct
  strict entrypoint is `utils.task_verification`; stock AXPBY defaults stay unchanged.
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
python -m scripts.validate --suite layernorm_residual_gelu --device cuda --purpose eval \
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
