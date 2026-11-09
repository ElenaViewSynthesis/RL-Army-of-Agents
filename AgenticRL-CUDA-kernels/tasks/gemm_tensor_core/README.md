# `gemm_tensor_core`

## Operator contract

A @ B + bias[N], fp16/bf16 operands with fp32 accumulation and fp32 output

Inputs are `A:(M,K)`, `B:(K,N)`, and fp32 `bias:(N,)`; output is fp32, not a bf16-rounded GEMM converted to fp32 afterward. Triton retains bf16/fp16 dot operands and accumulates in fp32. The CUDA fixture uses explicit WMMA/mma Tensor Core work with double-buffered shared-memory tiles. CuTe DSL must use MMA/copy atoms, not CUTLASS utility GEMM templates. The honest fixture acceptance target is Triton **at least 60% of the recorded cuBLAS throughput at bf16 4096^3 on H200**, with authored non-memory provenance **exactly 1.0**. This is unaccepted until measured. Skinny M=8 and M=64 cases are included. Candidates may not call cuBLAS.

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- Meta-PyTorch TritonBench: https://github.com/meta-pytorch/tritonbench; license: [BSD-3-Clause](https://github.com/meta-pytorch/tritonbench/blob/main/LICENSE).
- Official Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/index.html; license: [MIT](https://github.com/triton-lang/triton/blob/main/LICENSE).

Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
| `m512_n512_k512_bfloat16` | `bfloat16` | `{'m': 512, 'n': 512, 'k': 512}` |
| `m1024_n1024_k1024_bfloat16` | `bfloat16` | `{'m': 1024, 'n': 1024, 'k': 1024}` |
| `m2048_n2048_k2048_bfloat16` | `bfloat16` | `{'m': 2048, 'n': 2048, 'k': 2048}` |
| `m4096_n4096_k4096_bfloat16` | `bfloat16` | `{'m': 4096, 'n': 4096, 'k': 4096}` |
| `m512_n512_k512_float16` | `float16` | `{'m': 512, 'n': 512, 'k': 512}` |
| `m1024_n1024_k1024_float16` | `float16` | `{'m': 1024, 'n': 1024, 'k': 1024}` |
| `m2048_n2048_k2048_float16` | `float16` | `{'m': 2048, 'n': 2048, 'k': 2048}` |
| `m4096_n4096_k4096_float16` | `float16` | `{'m': 4096, 'n': 4096, 'k': 4096}` |
| `m8_n4096_k4096_bfloat16` | `bfloat16` | `{'m': 8, 'n': 4096, 'k': 4096}` |
| `m64_n4096_k4096_bfloat16` | `bfloat16` | `{'m': 64, 'n': 4096, 'k': 4096}` |
| `m8_n4096_k4096_float16` | `float16` | `{'m': 8, 'n': 4096, 'k': 4096}` |
| `m64_n4096_k4096_float16` | `float16` | `{'m': 64, 'n': 4096, 'k': 4096}` |

## Fixtures and gates

- Honest backends: triton, cuda, cutedsl.
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
python -m scripts.validate --suite gemm_tensor_core --device cuda --purpose eval \
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

Measured run: `task-stage23-20261109T013032Z`. Runtime (GPU, capability, UUID, Torch, Triton): `NVIDIA H200, 9.0, GPU-cf53da05-285c-0da7-97a9-43dc1b38646b, 2.14.1+cu130, 3.8.0`. Frozen source: `artifacts/task-stage23-20261109T013032Z/source`.

Recorded command (use a **new** output directory when reproducing):

```bash
python -m scripts.validate --suite prefix_sum,gemm_tensor_core,rmsnorm,log_softmax_gather,fused_adamw,layernorm_residual_gelu,grpo_group_norm,cispo_clip_weight,packed_varlen_reduction --device cuda --purpose eval --case-set smoke --fixtures honest,hack,wrong --timeout-s 600 --suite-out /artifacts/task-stage23-20261109T013032Z/suite --backends cuda,triton --benchmark --cold-start
```

Warm columns are CUDA-event medians, not reward-profiler sums. Cold is a single first-forward sample after initialization, not a cold-cache median. Rates are recorded workload diagnostics; logical GB/s is not HBM traffic.

| Case | Backend | Eager warm us | Compile warm us | Generated warm us | Generated first-forward us | Generated TFLOP/s | Generated logical GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `m512_n512_k512_bfloat16` | `triton` | 36.718400 | 63.899198 | 38.632001 | 748480.102539 | 6.948526 | 54.338371 |

Separately measured **cuBLAS GEMM-only with fp32 output**; generated includes bias. These paired samples are distinct from the warm table above. This is neither a CUTLASS-profiler nor a FlashAttention baseline.

| Case | Backend | Provider | cuBLAS us | Generated+bias us | cuBLAS TFLOP/s | Generated TFLOP/s | Percent of cuBLAS |
|---|---|---|---:|---:|---:|---:|---:|
| `m512_n512_k512_bfloat16` | `triton` | `_BlasBackend.Cublas` | 11.243200 | 31.993601 | 23.875361 | 8.390286 | 35.14% |

Captured measurement limits:
- Smoke cases only, not full grids or flagship acceptance
- All three gemm_tensor_core CUDA fixtures failed compilation; negative controls are NOT credited as passed
- Typed-zero CUDA fix is newer source and awaits a separate GPU retest
- CuTe DSL and advanced tasks are outside this run
- Benchmark and cuBLAS paired samples are diagnostics, not reward inputs
- No CUTLASS or FlashAttention vendor ceiling was measured

No absent timing, TFLOP/s, bandwidth, ceiling percentage, or H200 acceptance is zero-filled.
