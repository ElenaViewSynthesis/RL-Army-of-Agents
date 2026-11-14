# `rotary_embedding`

## Operator contract

Half-split RoPE on q [B,32,S,D] and k [B,8,S,D] with fp32 cos/sin [S,D/2]; returns new (q_rot, k_rot)

Stage 7, adapted from NVIDIA TileGym v1.5.1 `ops/cutile/rope.py`. Half-split (Llama) layout: `x1=x[...,:D/2]`, `x2=x[...,D/2:]`, outputs `x1*cos-x2*sin` and `x2*cos+x1*sin`. `q` is `(B,32,S,D)` and `k` is `(B,8,S,D)` (GQA); `cos`/`sin` are fp32 `(S,D/2)` Llama tables with base 10,000. **Outputs are new tensors**: TileGym rotates Q and K in place, which this contract forbids. The verifier compares outputs only and does not detect in-place mutation; that rule is enforced by source tests. Arithmetic is fp32 with one final cast. Acceptance is the last (largest) case.

### cuTile source candidate (opt-in)

One launch per tensor over 16-row tiles of the flattened (B*H*S, D) view, loading both halves with ct.load and the matching 16-row slice of the (S, D/2) tables; fp32 rotation, stores into a fresh output. S is a multiple of 16, so a tile never straddles a sequence boundary. cuTile timing, speedup and provenance are **not measured**: no GPU execution has validated these task kernels. Select `--backends cutile --fixtures all` explicitly; default backend rows are unchanged.

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
| `b1_hq32_hkv8_s512_d64_float32` | `float32` | `{'b': 1, 'hq': 32, 'hkv': 8, 's': 512, 'd': 64}` |
| `b1_hq32_hkv8_s512_d128_float32` | `float32` | `{'b': 1, 'hq': 32, 'hkv': 8, 's': 512, 'd': 128}` |
| `b4_hq32_hkv8_s2048_d64_float32` | `float32` | `{'b': 4, 'hq': 32, 'hkv': 8, 's': 2048, 'd': 64}` |
| `b4_hq32_hkv8_s2048_d128_float32` | `float32` | `{'b': 4, 'hq': 32, 'hkv': 8, 's': 2048, 'd': 128}` |
| `b8_hq32_hkv8_s8192_d64_float32` | `float32` | `{'b': 8, 'hq': 32, 'hkv': 8, 's': 8192, 'd': 64}` |
| `b8_hq32_hkv8_s8192_d128_float32` | `float32` | `{'b': 8, 'hq': 32, 'hkv': 8, 's': 8192, 'd': 128}` |
| `b1_hq32_hkv8_s512_d64_bfloat16` | `bfloat16` | `{'b': 1, 'hq': 32, 'hkv': 8, 's': 512, 'd': 64}` |
| `b1_hq32_hkv8_s512_d128_bfloat16` | `bfloat16` | `{'b': 1, 'hq': 32, 'hkv': 8, 's': 512, 'd': 128}` |
| `b4_hq32_hkv8_s2048_d64_bfloat16` | `bfloat16` | `{'b': 4, 'hq': 32, 'hkv': 8, 's': 2048, 'd': 64}` |
| `b4_hq32_hkv8_s2048_d128_bfloat16` | `bfloat16` | `{'b': 4, 'hq': 32, 'hkv': 8, 's': 2048, 'd': 128}` |
| `b8_hq32_hkv8_s8192_d64_bfloat16` | `bfloat16` | `{'b': 8, 'hq': 32, 'hkv': 8, 's': 8192, 'd': 64}` |
| `b8_hq32_hkv8_s8192_d128_bfloat16` | `bfloat16` | `{'b': 8, 'hq': 32, 'hkv': 8, 's': 8192, 'd': 128}` |

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
python -m scripts.validate --suite rotary_embedding --device cuda --purpose eval \
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
