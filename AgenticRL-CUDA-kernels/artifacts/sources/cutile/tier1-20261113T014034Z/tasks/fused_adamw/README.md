# `fused_adamw`

## Operator contract

Functional bias-corrected AdamW: return (new_parameter,new_m,new_v), never mutate inputs

Forward returns `(updated_parameter,new_m,new_v)` without mutating any input. The constructor supplies learning rate, beta1/beta2, epsilon, weight decay and step; bias correction is explicit. This is a functional numerical kernel benchmark, not a trainer or an integrated optimizer.

### cuTile source candidate (opt-in)

1024-element flattened tiles; one launch writes all three fp32 outputs. Bias corrections are scalar hyperparameter arithmetic; every input remains read-only. cuTile timing, speedup and provenance are **not measured**: no GPU execution has validated these task kernels. Select `--backends cutile --fixtures all` explicitly; default backend rows are unchanged.

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- Original AgenticRL workload derived from local GRPO/CISPO configuration: https://github.com/ElenaViewSynthesis/RL-Army-of-Agents; license: NOASSERTION: repository has no root license file.

Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
| `n4096_float32` | `float32` | `{'n': 4096}` |
| `n1048576_float32` | `float32` | `{'n': 1048576}` |
| `n8388608_float32` | `float32` | `{'n': 8388608}` |

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
python -m scripts.validate --suite fused_adamw --device cuda --purpose eval \
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
| `n4096_float32` | `cuda` | 88.414398 | 36.867200 | 13.531200 | 4023.456097 | unavailable | 8.475819 |
| `n4096_float32` | `triton` | 99.129599 | 40.859200 | 31.080000 | 128055.099487 | unavailable | 3.690090 |

Captured measurement limits:
- Smoke cases only, not full grids or flagship acceptance
- All three gemm_tensor_core CUDA fixtures failed compilation; negative controls are NOT credited as passed
- Typed-zero CUDA fix is newer source and awaits a separate GPU retest
- CuTe DSL and advanced tasks are outside this run
- Benchmark and cuBLAS paired samples are diagnostics, not reward inputs
- No CUTLASS or FlashAttention vendor ceiling was measured

No absent timing, TFLOP/s, bandwidth, ceiling percentage, or H200 acceptance is zero-filled.
