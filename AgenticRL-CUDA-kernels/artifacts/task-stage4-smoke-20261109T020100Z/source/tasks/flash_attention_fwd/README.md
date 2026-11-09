# `flash_attention_fwd`

## Operator contract

Causal softmax(Q @ K.T / sqrt(64)) @ V; fp32 attention arithmetic, bf16 output; tolerance 1e-2

The constructor is `(causal=True)`. `get_inputs()` returns `(Q,K,V)` with `B=4,H=8,d=64,S in {512,1024,2048}` and bf16 inputs/output. Reference QK, softmax and PV arithmetic is fp32. Tolerance stays **atol=rtol=1e-2**; 2e-2 was not authorized. The honest Triton path uses tiled online softmax; CUDA uses a two-pass shared-memory implementation. CuTe DSL MMA/online-softmax and TileLang T.gemm/pipelined-softmax are stretch backends, not implicitly passed fixtures. The deliberate reward hack calls PyTorch scaled_dot_product_attention. Algorithmic causal FLOPs are `2*B*H*S*(S+1)*d`, approximately half the full `4*B*H*S*S*d`; these are not measured executed-instruction counts. A baseline must identify its actual Flash/SDPA provider/version before a ceiling percentage is reported; an unverified provider must not be relabeled FA2.

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
| `b4_h8_s512_d64_bfloat16` | `bfloat16` | `{'b': 4, 'h': 8, 's': 512, 'd': 64}` |
| `b4_h8_s1024_d64_bfloat16` | `bfloat16` | `{'b': 4, 'h': 8, 's': 1024, 'd': 64}` |
| `b4_h8_s2048_d64_bfloat16` | `bfloat16` | `{'b': 4, 'h': 8, 's': 2048, 'd': 64}` |

## Fixtures and gates

- Honest backends: triton, cuda.
- Stretch backends: cutedsl, tilelang.
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
python -m scripts.validate --suite flash_attention_fwd --device cuda --purpose eval \
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
