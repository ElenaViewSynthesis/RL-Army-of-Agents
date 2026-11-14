# `geglu_tanh`

## Operator contract

x [B,2H] -> x[:, :H] * gelu_tanh(x[:, H:]) in fp32 (tanh GELU on the second half); output [B,H] input dtype

Stage 7, adapted from NVIDIA TileGym v1.5.1 `ops/cutile/activation/geglu.py` and `gelu.py`. The input is `(B,2H)`; **the first half is the multiplier and GELU applies to the second half**, the opposite split from `silu_and_mul`. GELU is the **tanh approximation** `0.5*b*(1+tanh(sqrt(2/pi)*(b+0.044715*b^3)))`; TileGym's `approximate="none"` path uses the same tanh formula, and cuTile 1.8.0 has no `erf`. Tanh and exact erf GELU differ by at most 4.7e-4, inside the tolerance, so this task is **not** evidence about the exact-erf `layernorm_residual_gelu` contract. Arithmetic is fp32 with one final cast. Acceptance is the last (largest) case.

### cuTile source candidate (opt-in)

Two-dimensional grid (row, 1024-wide chunk); the multiplier from the first half, tanh-GELU of the second half computed as 1-2/(exp(2z)+1) in fp32 (exp is H200-validated; ct.tanh is not), input-dtype output. cuTile timing, speedup and provenance are **not measured**: no GPU execution has validated these task kernels. Select `--backends cutile --fixtures all` explicitly; default backend rows are unchanged.

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- NVIDIA TileGym v1.5.1 (commit bb1a7e720594880be017e8fdf8f7e4f740656beb): https://github.com/NVIDIA/TileGym; license: [MIT (LICENSE Section A; kernel sources only)](https://github.com/NVIDIA/TileGym/blob/bb1a7e720594880be017e8fdf8f7e4f740656beb/LICENSE).

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
| `b64_h16384_float32` | `float32` | `{'b': 64, 'h': 16384}` |
| `b4096_h1024_float32` | `float32` | `{'b': 4096, 'h': 1024}` |
| `b4096_h4096_float32` | `float32` | `{'b': 4096, 'h': 4096}` |
| `b4096_h16384_float32` | `float32` | `{'b': 4096, 'h': 16384}` |
| `b64_h1024_bfloat16` | `bfloat16` | `{'b': 64, 'h': 1024}` |
| `b64_h4096_bfloat16` | `bfloat16` | `{'b': 64, 'h': 4096}` |
| `b64_h16384_bfloat16` | `bfloat16` | `{'b': 64, 'h': 16384}` |
| `b4096_h1024_bfloat16` | `bfloat16` | `{'b': 4096, 'h': 1024}` |
| `b4096_h4096_bfloat16` | `bfloat16` | `{'b': 4096, 'h': 4096}` |
| `b4096_h16384_bfloat16` | `bfloat16` | `{'b': 4096, 'h': 16384}` |

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
python -m scripts.validate --suite geglu_tanh --device cuda --purpose eval \
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
