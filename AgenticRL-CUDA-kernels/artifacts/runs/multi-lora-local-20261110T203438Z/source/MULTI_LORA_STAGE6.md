# Stage 6: standalone multi-LoRA contracts

Six engine-free tasks implement the shrink/expand equations in
MULTI_LORA_kernels_task.md. No serving framework, TP collective, custom-op
registration or base-weight format is involved. Inputs are contiguous; all
constructors/state dictionaries are empty. Inputs are never mutated.

## Sources

Authored from the operation descriptions, with no upstream kernel code copied.
Punica, S-LoRA and vLLM each publish Apache-2.0. Retrieved licenses, pinned
revisions, retrieval times and SHA-256 digests are in `legal/multi_lora/`.
`utils/task_catalog.py::SOURCES` records each origin. Stage 6 is not marked
evaluation-only; these upstream licenses do not supply a blanket license for
this repository, which has no root LICENSE.

## Interfaces

- `lora_shrink_bgmv`: `(x,A,idx,ranks,scales)` → fp32 `buf[S,T,R]`.
- `lora_expand_bgmv`: `(buf,B,y_in,idx,ranks,offsets)` → output matching y_in.
- `lora_segment_metadata`: `(idx,slots)` → `(order,ids,starts,lengths,active)`.
  `slots` is an int32 vector of length max_loras; only its shape is used.
- `lora_shrink_sgmv`: shrink arguments followed by the five metadata tensors.
- `lora_expand_sgmv`: expand arguments followed by the five metadata tensors.
- `lora_fused_sgmv`: `(x,A,B,y_in,idx,ranks,scales,offsets,*metadata)` → y.
  This is the allowed fp32-scratch implementation, with two arithmetic passes,
  rather than a claim that both passes remain in registers.

A has layout `[L,S,R,H]`; B `[L,S,N,R]`. R is rank rounded up to 16.
Ranks and IDs are int32, scales fp32; A/B are cast once in the trusted builder.
Each slice has N outputs with offset `8+s*(N+16)`; y width is `S*(N+16)`.
These disjoint slices have leading/trailing guard columns. All guards and skipped
token rows are preserved exactly. Functional output initialization copies y_in;
no LoRA update is written to a skipped token. Shrink returns canonical zero for
inactive tokens and padded ranks. Ranks/scales/offsets are device inputs, not
Python constants. Every multiply and accumulation uses fp32; only the final
addition into y is downcast. A/B rank padding and expand-input buffer padding
are NaN. Kernels must mask before loads/arithmetic, not multiply NaNs by zero.

Metadata order is stable ascending adapter slot, then original token position.
Skipped tokens follow all active tokens, in original order. `order[T]` is a
permutation including skips. `ids/starts/lengths[L]` hold only nonempty active
segments, compacted in ascending slot order. Inactive entries are `(-1,0,0)`.
`active[1]` counts active segments. All five outputs are int32 and verified
exactly. All-skipped input returns order 0..T-1 and zero active segments.

The reference uses static slot/slice loops over vectorized fp32 Torch matmuls
and masks, with no per-token host loop or tensor-dependent host synchronization.
Reference model construction selects highest float32 matmul precision, matching
the existing GEMM references; TF32 is not substituted for fp32 products.
This bounds memory independently of T copies of the adapter matrices. It does
extra work for inactive slots; this is an eager reference, not an optimized
serving baseline. The independent token-loop oracle lives only in tests.

## Implementations and cases

Triton BGMV shrink splits hidden K into fixed 1024-wide reductions and merges
fp32 partials; expand uses 16-output tiles. SGMV uses fixed 16x16x32 fp32
`tl.dot(input_precision="ieee")` tiles, reusing weights across a segment's token
tile. This avoids bf16 rounding of the fp32 intermediate and TF32 assumptions
on sm_75. No broad autotuning or vendor GEMM is used. The CUDA counterpart is a
conservative correctness implementation with block reductions/explicit rank
loops; segment programs reuse metadata but do not yet share matrix tiles.
CUDA instantiates native `__half`/`__nv_bfloat16` device signatures with explicit
conversion intrinsics: Torch `c10::Half` symbols are conservatively rejected by
the existing classifier. No attribution rule is relaxed.
All flattened offsets use int64 and large dimensions use grid axis 0. Metadata
uses counts followed by stable per-slot scans/scatters; CUDA uses a simpler
quadratic placement oracle on device. Performance is unmeasured.

Smoke is the smallest declared case per task: one token for BGMV/metadata,
512 tokens for SGMV. The small three-token mixed BGMV case exercises skip and
padding controls together.

The paired, deterministic catalog covers hidden/output dimensions
2048/4096/5120/8192/14336, all ranks 8/16/32/64/128/256, slots 1/4/16,
slices 1/2/3 and fp16/bf16. It covers mixed ranks and single, distinct,
round-robin, mixed-skipped and all-skipped assignments. `stratified` selects
this entire small paired grid, not a Cartesian product. BGMV stops at 256
tokens. SGMV uses 512 and 4096 tokens; acceptance selects the final
4096-token, 14336-hidden/output, rank-256, 16-slot, three-slice bf16 case.
Rank 512 and 8192-token shapes remain stretch extensions.

## Controls and diagnostics

Every task has honest Triton/CUDA, Torch-op hack and wrong-output fixtures.
Arithmetic tasks also have `wrong_padding` and `wrong_skip`. Cases with no rank
padding or no skipped tokens report the corresponding control as unavailable;
there is no observable violation to test in those cases. Metadata has no weight
padding, buffer precision or y output, so those arithmetic controls do not apply.
`--fixtures all` discovers applicable wrong-padding/skip files.

`fp16_intermediate` files are diagnostic only; they are not accepted as suite
failure gates. For each verified honest arithmetic case, a separate diagnostic
process writes `lora-numerics.json` and embeds its distributions in
`suite-case.json`: max/p50/p99 absolute error, max-error/max-reference ratio,
and the original dtype-specific bound (.01 bf16/.005 fp16). Honest and half
intermediate errors appear side by side. The approved relative rule below gates
Stage 6 arithmetic verification; the fp16-intermediate comparison stays diagnostic.
At rank below 64 this comparison is descriptive only; the rank≥64 catalog cases
provide the requested accuracy study when executed. No error numbers are
invented for unexecuted cases.

Workload rates count only active ranks/tokens. Shrink bytes count active x,
distinct touched A rows and active fp32 buffers; expand FLOPs count twice each
active rank/output multiply-add. These are logical rates, not measured HBM
traffic. Masked padding is excluded. Current y-copy and scratch implementation
traffic is not included in the logical rate. Cold/warm and graph measurements
are diagnostics only.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
# After explicit authorization, on a GPU host, freeze public source first:
python -m scripts.freeze_task_suite --tasks stage6 --output artifacts/runs/stage6-source-UNIQUE
# Run from the frozen source using the repository's GPU runner:
python -m scripts.validate --suite stage6 --device cuda --purpose eval \
  --case-set smoke --fixtures all --benchmark --cold-start --cuda-graphs \
  --timeout-s 600 --suite-out artifacts/runs/stage6-smoke-UNIQUE
# Only after smoke review, use --case-set acceptance with another unique output.
```

Both CUDA compilation and Triton real-device execution need GPU validation.
CPU interpreter execution is not GPU provenance, sm_75 support, graph replay,
latency or H200 acceptance evidence. No Modal run is launched by this change.


## Approved verification and replay repair (2026-11-10)

`RolloutConfig.relative_error_bounds` defaults off. The progressive suite enables
it only for Stage 6 arithmetic and selects `utils.lora_verification` in the same
verification subprocess stage. For each float output the maximum absolute error
must be at most 0.005 times the checked reference maximum (fp16 activations), or
0.01 (bf16 activations), even when shrink returns fp32. Skipped y rows are checked
bit-for-bit against y_in and excluded from reductions; zero reference requires
exact equality. Non-finite errors fail; integers retain exact verification.
This replaces the prior per-element floating check, without changing the stock
or Stage 0–5 paths, speed reward or provenance semantics. Detailed gate decisions
are recorded in `.cache/relative-error-bounds.json` and `suite-case.json`.
Diagnostics include the half intermediate's hypothetical decision but do not
turn it into a failure fixture or acceptance count.

The benchmark capture helper now returns an owning `CapturedGraph` handle that
keeps static input clones and captured outputs alive for all timed replays.
A CPU weak-reference repro fails on the archived helper and passes on the fix.
Repeated CPU reference/Triton calls preserve every input byte for 40 iterations;
no caller mutation was found in these probes. Native CUDA paths and the actual
H200 crash resolution require another approved GPU run. Torch graph-pool output
allocations remain supported; this fix preserves buffers and does not change
forward signatures or introduce in-place y updates. See the PyTorch 2.14
[graph lifetime and allocator contract](https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management).
