# `prefix_sum`

## Operator contract

Global inclusive scan, not independent block scans; nonnegative inputs; int32 does not overflow

The scan is globally inclusive across block boundaries. Integer inputs are in `[0,3]`, so the declared largest shape cannot overflow int32. Floating inputs are nonnegative fp32 in `[0,1)`. The multi-pass implementation recursively scans block totals and then adds offsets; independent block scans are incorrect.

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- GPU MODE reference-kernels (evaluation-only adaptation): https://github.com/gpu-mode/reference-kernels; license: [June 9 Researcher Reciprocity License 1.0](https://github.com/gpu-mode/reference-kernels/blob/main/LICENSE).

**Held-out evaluation only.** GPU MODE's June 9 Researcher Reciprocity License covers evaluation/benchmarking for training as Training Use. This modified problem-family adaptation is blocked from `purpose=train` until reciprocity is cleared. Labeling an RL feedback run 'eval' does not clear those obligations.

Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
| `n100000_int32` | `int32` | `{'n': 100000}` |
| `n1000000_int32` | `int32` | `{'n': 1000000}` |
| `n10000000_int32` | `int32` | `{'n': 10000000}` |
| `n100000_float32` | `float32` | `{'n': 100000}` |
| `n1000000_float32` | `float32` | `{'n': 1000000}` |
| `n10000000_float32` | `float32` | `{'n': 10000000}` |

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
python -m scripts.validate --suite prefix_sum --device cuda --purpose eval \
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
| `n100000_int32` | `cuda` | 15.531199 | 57.017601 | 39.975999 | 13462.528229 | unavailable | 20.012008 |
| `n100000_int32` | `triton` | 13.580801 | 51.390401 | 56.521600 | 109617.538452 | unavailable | 14.153881 |

Captured measurement limits:
- Smoke case only, not full grids or flagship performance gates
- Integer outputs used the stock 1e-2 comparison in this historical run
- New exact-integer policy requires a new GPU run
- Benchmark samples and rates are diagnostics, not reward inputs
- No CUTLASS or FlashAttention ceiling was measured

### Exact-integer H200 revalidation

Measured run: `task-stage23-20261109T013032Z`. Runtime (GPU, capability, UUID, Torch, Triton): `NVIDIA H200, 9.0, GPU-cf53da05-285c-0da7-97a9-43dc1b38646b, 2.14.1+cu130, 3.8.0`. Frozen source: `artifacts/task-stage23-20261109T013032Z/source`.

Recorded command (use a **new** output directory when reproducing):

```bash
python -m scripts.validate --suite prefix_sum,gemm_tensor_core,rmsnorm,log_softmax_gather,fused_adamw,layernorm_residual_gelu,grpo_group_norm,cispo_clip_weight,packed_varlen_reduction --device cuda --purpose eval --case-set smoke --fixtures honest,hack,wrong --timeout-s 600 --suite-out /artifacts/task-stage23-20261109T013032Z/suite --backends cuda,triton --benchmark --cold-start
```

Warm columns are CUDA-event medians, not reward-profiler sums. Cold is a single first-forward sample after initialization, not a cold-cache median. Rates are recorded workload diagnostics; logical GB/s is not HBM traffic.

| Case | Backend | Eager warm us | Compile warm us | Generated warm us | Generated first-forward us | Generated TFLOP/s | Generated logical GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `n100000_int32` | `cuda` | 12.942400 | 53.340802 | 35.495999 | 8809.568405 | unavailable | 22.537751 |
| `n100000_int32` | `triton` | 13.056000 | 40.540799 | 49.382401 | 118045.761108 | unavailable | 16.200104 |

Captured measurement limits:
- Smoke cases only, not full grids or flagship acceptance
- All three gemm_tensor_core CUDA fixtures failed compilation; negative controls are NOT credited as passed
- Typed-zero CUDA fix is newer source and awaits a separate GPU retest
- CuTe DSL and advanced tasks are outside this run
- Benchmark and cuBLAS paired samples are diagnostics, not reward inputs
- No CUTLASS or FlashAttention vendor ceiling was measured

No absent timing, TFLOP/s, bandwidth, ceiling percentage, or H200 acceptance is zero-filled.
