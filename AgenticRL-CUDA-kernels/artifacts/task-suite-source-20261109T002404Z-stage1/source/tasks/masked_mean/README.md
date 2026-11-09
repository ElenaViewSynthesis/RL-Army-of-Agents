# `masked_mean`

## Operator contract

Mean of x[row,:lengths[row]]; empty rows return fp32 zero; this is not trimmed mean

Lengths are int32 and satisfy `0 <= length <= L`. At least one generated row is empty and one is full. Empty rows return zero. This is a masked/prefix mean, not an order-statistic trimmed mean. It is a modified problem-family adaptation, not a claim to reproduce a particular GPU MODE competition submission.

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
| `b64_l128_float32` | `float32` | `{'b': 64, 'l': 128}` |
| `b64_l256_float32` | `float32` | `{'b': 64, 'l': 256}` |
| `b64_l512_float32` | `float32` | `{'b': 64, 'l': 512}` |
| `b64_l1024_float32` | `float32` | `{'b': 64, 'l': 1024}` |
| `b64_l2048_float32` | `float32` | `{'b': 64, 'l': 2048}` |
| `b64_l4096_float32` | `float32` | `{'b': 64, 'l': 4096}` |
| `b64_l8192_float32` | `float32` | `{'b': 64, 'l': 8192}` |
| `b512_l128_float32` | `float32` | `{'b': 512, 'l': 128}` |
| `b512_l256_float32` | `float32` | `{'b': 512, 'l': 256}` |
| `b512_l512_float32` | `float32` | `{'b': 512, 'l': 512}` |
| `b512_l1024_float32` | `float32` | `{'b': 512, 'l': 1024}` |
| `b512_l2048_float32` | `float32` | `{'b': 512, 'l': 2048}` |
| `b512_l4096_float32` | `float32` | `{'b': 512, 'l': 4096}` |
| `b512_l8192_float32` | `float32` | `{'b': 512, 'l': 8192}` |
| `b64_l128_bfloat16` | `bfloat16` | `{'b': 64, 'l': 128}` |
| `b64_l256_bfloat16` | `bfloat16` | `{'b': 64, 'l': 256}` |
| `b64_l512_bfloat16` | `bfloat16` | `{'b': 64, 'l': 512}` |
| `b64_l1024_bfloat16` | `bfloat16` | `{'b': 64, 'l': 1024}` |
| `b64_l2048_bfloat16` | `bfloat16` | `{'b': 64, 'l': 2048}` |
| `b64_l4096_bfloat16` | `bfloat16` | `{'b': 64, 'l': 4096}` |
| `b64_l8192_bfloat16` | `bfloat16` | `{'b': 64, 'l': 8192}` |
| `b512_l128_bfloat16` | `bfloat16` | `{'b': 512, 'l': 128}` |
| `b512_l256_bfloat16` | `bfloat16` | `{'b': 512, 'l': 256}` |
| `b512_l512_bfloat16` | `bfloat16` | `{'b': 512, 'l': 512}` |
| `b512_l1024_bfloat16` | `bfloat16` | `{'b': 512, 'l': 1024}` |
| `b512_l2048_bfloat16` | `bfloat16` | `{'b': 512, 'l': 2048}` |
| `b512_l4096_bfloat16` | `bfloat16` | `{'b': 512, 'l': 4096}` |
| `b512_l8192_bfloat16` | `bfloat16` | `{'b': 512, 'l': 8192}` |

## Fixtures and gates

- Honest backends: triton, cuda.
- Stretch backends: none declared.
- `model_new.py` selects the default honest fixture. `model_new_hack.py` is an
  intentional Torch reference-compute attack; it must verify then fail provenance.
- `model_new_wrong.py` intentionally returns NaNs or invalid integer sentinels;
  it must fail verification, not merely compilation.
- Verification uses five fresh random inputs and atol=rtol=1e-2.
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
python -m scripts.validate --suite masked_mean --device cuda --purpose eval \
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
