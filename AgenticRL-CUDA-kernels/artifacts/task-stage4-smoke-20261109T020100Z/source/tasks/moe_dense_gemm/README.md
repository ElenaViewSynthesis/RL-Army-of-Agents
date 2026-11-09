# `moe_dense_gemm`

## Operator contract

Dense all-expert FC1, fp32 SwiGLU and alpha mask, bf16 intermediate, dense FC2; fp32 output


### Dense-MoE BF16 adaptation

This is a methodology adaptation of the linked Blackwell article, not a port of
TensorRT-LLM's prewritten kernels. With E routed experts and optional one shared
expert, inputs are X `(T,H)`, gate/up weights `(E_total,H,I)`, down weights
`(E_total,I,H)`, and fp32 alpha `(T,E_total)`. Alpha is zero for unselected routed
experts, normalized positive weights for selected experts, and one for the optional
shared expert. All expert FC1 columns and FC2 K blocks are computed densely:

`FC1 = X @ concat(gate,up)` with fp32 accumulation; then
`A = (SiLU(gate_result) * up_result * alpha).to(bf16)`; then
`O = A.flatten(experts,I) @ down.flatten(experts,I)` with fp32 accumulation/output.

Intermediate I is a multiple of 256. This implementation's logical gate/up packing
is explicit in model.py; the article's 64-column interleave and NVFP4 scale swizzles
are not claimed to be the H200 layout. Dense-GEMM FLOP/s counts all experts, not just
the routed top-k. Do not compare those counts with useful-routed FLOPs as if they
were the same work. No Blackwell speedup from the article is a repository baseline.

### NVFP4/Blackwell contract (not implemented on H200)

This separate future precision contract requires Blackwell NVFP4 MMA support.
Shapes and routing math follow the BF16 contract, with I 256-aligned. Logical
FP4 values are E2M1 with magnitudes `{0,0.5,1,1.5,2,3,4,6}`, two signed nibbles
per byte, increasing logical index in the low nibble first. Each contiguous
16-value K block has a positive E4M3FN scale and each tensor a positive fp32
global scale; the logical dequantization is
`decode_fp4(nibble) * decode_e4m3(block_scale) * global_scale`.

The quantization-aware reference must dequantize X and all weights, perform
dense FC1 in fp32, apply SwiGLU, requantize the activation in the same block
format using round-to-nearest-even with saturation to magnitude 6, then
dequantize it for FC2. Activation block scales are explicit inputs to this
contract, not a hidden calibration algorithm. Routing alpha is applied per
token/expert before the corresponding FC2 K block; unselected experts have
alpha zero. W2 logical storage is hidden-major `(H,E_total*I)`; physical scale
swizzles must be separately specified by a future hardware implementation.

There is no NVFP4 candidate, performance number, native H200 support, or accepted
precision-tolerance claim. Existing 1e-2 tolerance is not silently relaxed.


The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

- TensorRT-LLM MoE as Dense GEMM (BF16 H200 adaptation): https://nvidia.github.io/TensorRT-LLM/1.3.0rc15/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html; license: [Apache-2.0 (TensorRT-LLM project); article attribution retained](https://github.com/NVIDIA/TensorRT-LLM/blob/main/LICENSE).
- Official Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/index.html; license: [MIT](https://github.com/triton-lang/triton/blob/main/LICENSE).

Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
| `t8_h256_i512_e4_top_k2_sharedfalse_bfloat16` | `bfloat16` | `{'t': 8, 'h': 256, 'i': 512, 'e': 4, 'top_k': 2, 'shared': False}` |
| `t32_h256_i512_e4_top_k2_sharedfalse_bfloat16` | `bfloat16` | `{'t': 32, 'h': 256, 'i': 512, 'e': 4, 'top_k': 2, 'shared': False}` |
| `t64_h256_i512_e4_top_k2_sharedfalse_bfloat16` | `bfloat16` | `{'t': 64, 'h': 256, 'i': 512, 'e': 4, 'top_k': 2, 'shared': False}` |
| `t128_h256_i512_e4_top_k2_sharedfalse_bfloat16` | `bfloat16` | `{'t': 128, 'h': 256, 'i': 512, 'e': 4, 'top_k': 2, 'shared': False}` |
| `t64_h512_i512_e8_top_k2_sharedfalse_bfloat16` | `bfloat16` | `{'t': 64, 'h': 512, 'i': 512, 'e': 8, 'top_k': 2, 'shared': False}` |
| `t64_h512_i512_e8_top_k2_sharedtrue_bfloat16` | `bfloat16` | `{'t': 64, 'h': 512, 'i': 512, 'e': 8, 'top_k': 2, 'shared': True}` |

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
python -m scripts.validate --suite moe_dense_gemm --device cuda --purpose eval \
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
