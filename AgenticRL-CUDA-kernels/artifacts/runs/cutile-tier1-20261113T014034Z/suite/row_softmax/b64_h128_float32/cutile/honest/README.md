# `row_softmax`

## Operator contract

Stable row softmax with fp32 accumulation; output input dtype



### cuTile source candidate (opt-in)

One block per row, power-of-two tile up to 8192; negative-infinity gather padding for max, fp32 exp/sum, input-dtype output. Out-of-range stores are skipped. cuTile timing, speedup and provenance are **not measured**: no GPU execution has validated these task kernels. Select `--backends cutile --fixtures all` explicitly; default backend rows are unchanged.

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
| `b64_h128_float32` | `float32` | `{'b': 64, 'h': 128}` |
| `b64_h1024_float32` | `float32` | `{'b': 64, 'h': 1024}` |
| `b64_h8192_float32` | `float32` | `{'b': 64, 'h': 8192}` |
| `b512_h128_float32` | `float32` | `{'b': 512, 'h': 128}` |
| `b512_h1024_float32` | `float32` | `{'b': 512, 'h': 1024}` |
| `b512_h8192_float32` | `float32` | `{'b': 512, 'h': 8192}` |
| `b64_h128_float16` | `float16` | `{'b': 64, 'h': 128}` |
| `b64_h1024_float16` | `float16` | `{'b': 64, 'h': 1024}` |
| `b64_h8192_float16` | `float16` | `{'b': 64, 'h': 8192}` |
| `b512_h128_float16` | `float16` | `{'b': 512, 'h': 128}` |
| `b512_h1024_float16` | `float16` | `{'b': 512, 'h': 1024}` |
| `b512_h8192_float16` | `float16` | `{'b': 512, 'h': 8192}` |
| `b64_h128_bfloat16` | `bfloat16` | `{'b': 64, 'h': 128}` |
| `b64_h1024_bfloat16` | `bfloat16` | `{'b': 64, 'h': 1024}` |
| `b64_h8192_bfloat16` | `bfloat16` | `{'b': 64, 'h': 8192}` |
| `b512_h128_bfloat16` | `bfloat16` | `{'b': 512, 'h': 128}` |
| `b512_h1024_bfloat16` | `bfloat16` | `{'b': 512, 'h': 1024}` |
| `b512_h8192_bfloat16` | `bfloat16` | `{'b': 512, 'h': 8192}` |

## Fixtures and gates

- Honest backends: triton, cuda.
- Stretch backends: cutile.
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
  tl.dot, ct.mma and authored CuTe MMA/copy atoms are allowed.
- Name matching remains a heuristic, not proof of authorship or a security sandbox.
- CPU/missing capabilities are **unavailable**, never GPU passes.

## Reproduction and measured baselines

From the project root, on an H200 runtime:

```bash
python -m scripts.validate --suite row_softmax --device cuda --purpose eval \
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

Measured run: `task-stage1-20261109T004826Z`. Runtime (GPU, capability, UUID, Torch, Triton): `NVIDIA H200, 9.0, GPU-e1a0c72c-39d3-88be-ccc8-f846100efef1, 2.14.1+cu130, 3.8.0`. Frozen source: `artifacts/task-suite-source-20261109T002404Z-stage1/source`.

Recorded command (use a **new** output directory when reproducing):

```bash
python -m scripts.validate --suite gemm_div_sum_scale,row_softmax,masked_mean,prefix_sum --device cuda --purpose eval --case-set smoke --fixtures honest,hack,wrong --timeout-s 600 --suite-out /artifacts/task-stage1-20261109T004826Z/suite --backends cuda,triton --benchmark --cold-start
```

Warm columns are CUDA-event medians, not reward-profiler sums. Cold is a single first-forward sample after initialization, not a cold-cache median. Rates are recorded workload diagnostics; logical GB/s is not HBM traffic.

| Case | Backend | Eager warm us | Compile warm us | Generated warm us | Generated first-forward us | Generated TFLOP/s | Generated logical GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `b64_h128_float32` | `cuda` | 9.548800 | 33.329600 | 7.780800 | 1235.200047 | unavailable | 8.422784 |
| `b64_h128_float32` | `triton` | 11.572800 | 37.720001 | 23.123200 | 88838.562012 | unavailable | 2.834210 |

Captured measurement limits:
- Smoke case only, not full grids or flagship performance gates
- Integer outputs used the stock 1e-2 comparison in this historical run
- New exact-integer policy requires a new GPU run
- Benchmark samples and rates are diagnostics, not reward inputs
- No CUTLASS or FlashAttention ceiling was measured

No absent timing, TFLOP/s, bandwidth, ceiling percentage, or H200 acceptance is zero-filled.
