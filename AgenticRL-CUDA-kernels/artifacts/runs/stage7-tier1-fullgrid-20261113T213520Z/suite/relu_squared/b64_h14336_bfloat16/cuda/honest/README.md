# `relu_squared`

## Operator contract

relu(x.float())**2 elementwise on [B,H]; output input dtype; inputs read-only

Stage 7, adapted from NVIDIA TileGym v1.5.1 `suites/liger/cutile/relu_squared.py` (a cuTile port of the Liger-Kernel operator). Arithmetic is fp32; the output is cast once to the input dtype and is a new tensor. H=14,336 is not a multiple of 1,024, so the largest cases exercise the tail. Acceptance is the last (largest) case.

### cuTile source candidate (opt-in)

Two-dimensional grid (row, 1024-wide chunk) with bounds-checked gathers and scatters; fp32 max and square, input-dtype output. No tuning, capability branch or launch hint. cuTile timing, speedup and provenance are **not measured**: no GPU execution has validated these task kernels. Select `--backends cutile --fixtures all` explicitly; default backend rows are unchanged.

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- NVIDIA TileGym v1.5.1 (commit bb1a7e720594880be017e8fdf8f7e4f740656beb): https://github.com/NVIDIA/TileGym; license: [MIT (LICENSE Section A; kernel sources only)](https://github.com/NVIDIA/TileGym/blob/bb1a7e720594880be017e8fdf8f7e4f740656beb/LICENSE).
- Liger-Kernel (origin of TileGym's Liger-suite operator contracts): https://github.com/linkedin/Liger-Kernel; license: [BSD-2-Clause](https://github.com/linkedin/Liger-Kernel/blob/42dc23c41dcaf1f95d858837aed5301fd4faa27a/LICENSE).

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
| `b64_h14336_float32` | `float32` | `{'b': 64, 'h': 14336}` |
| `b4096_h1024_float32` | `float32` | `{'b': 4096, 'h': 1024}` |
| `b4096_h4096_float32` | `float32` | `{'b': 4096, 'h': 4096}` |
| `b4096_h14336_float32` | `float32` | `{'b': 4096, 'h': 14336}` |
| `b64_h1024_bfloat16` | `bfloat16` | `{'b': 64, 'h': 1024}` |
| `b64_h4096_bfloat16` | `bfloat16` | `{'b': 64, 'h': 4096}` |
| `b64_h14336_bfloat16` | `bfloat16` | `{'b': 64, 'h': 14336}` |
| `b4096_h1024_bfloat16` | `bfloat16` | `{'b': 4096, 'h': 1024}` |
| `b4096_h4096_bfloat16` | `bfloat16` | `{'b': 4096, 'h': 4096}` |
| `b4096_h14336_bfloat16` | `bfloat16` | `{'b': 4096, 'h': 14336}` |

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
python -m scripts.validate --suite relu_squared --device cuda --purpose eval \
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
