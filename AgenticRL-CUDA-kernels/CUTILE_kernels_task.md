# Task: cuTile kernels beyond AXPBY (honest cuTile solutions for existing catalog tasks)

You are working in AgenticRL-CUDA-kernels/ (WSL2). Read AGENTS.md, CLAUDE.md, the
binding Ground rules in Benchmark_diagnostics_task.md, SKILL_CUTILE.md, the README
section "TileLang, cuTile and CuTe DSL backends", and these files before writing
anything:

- `kernels/cutile/axpby.py` and `model_new_cutile.py`: the only cuTile kernel in
  the repository today (`axpby_forward`), validated on an H200.
- `tasks/gemm_tensor_core/`: the only task that already ships a third backend
  (`kernels/cutedsl/`, `model_new_cutedsl.py`, `backends=("triton", "cuda", "cutedsl")`
  in `utils/task_catalog.py`). Match this layout exactly.
- `utils/backend.py` (the `cutile` `BackendSpec`), `utils/task_suite.py`
  (`scaffold`, `evaluate_case`, `expected_outcome`) and `utils/task_policy.py`.

## Goal

`from kernels.cutile import axpby_forward` is the only cuTile entry point so far, so
the cuTile backend has been exercised on one 1 x 128 elementwise kernel. Add honest
cuTile solutions for the existing catalog tasks listed below, so that a task workdir
can do `from kernels.cutile import <task>_forward` and be scored by the unchanged
compile -> verify -> profile -> reward pipeline.

This adds a backend to existing tasks. It does **not** add catalog tasks, cases,
shapes, dtypes, tolerances, reward rules or provenance rules. `kernels/cutile/axpby.py`,
`kernels/cutile/__init__.py` and `model_new_cutile.py` at the repository root stay
byte-identical.

## Tasks, in order

Do them in this order and finish each tier (code, tests, docs) before the next.
Each tier exercises a different part of the cuTile API.

**Tier 1: row reductions and elementwise math**

1. `rmsnorm`: `ct.sum` of squares in fp32, `ct.rsqrt`, multiply by `weight[H]`, cast
   to the input dtype. Rows are up to 8,192 wide, so decide between one block per row
   with an in-kernel loop over tiles and a wider tile; document the choice.
2. `row_softmax`: `ct.max`, `ct.exp`, `ct.sum` with fp32 accumulation; three dtypes.
3. `cispo_clip_weight`: `ct.exp`, `ct.minimum`/`ct.maximum` (there is no `clip`),
   mask multiply.
4. `fused_adamw`: three outputs `(new_parameter, new_m, new_v)` from one kernel,
   never mutating inputs; `ct.sqrt`, 8,388,608 elements in the largest case.
5. `grpo_group_norm`: small population mean/std per group.

**Tier 2: indexed access**

6. `log_softmax_gather`: stable fp32 log-softmax over a vocabulary of up to 128,256,
   then one `ct.gather` per row by token index.
7. `residual_rmsnorm`: two outputs, bf16 residual sum plus fp32 RMSNorm accumulation.

**Tier 3: matrix multiply**

8. `gemm_div_sum_scale`: `ct.mma` with an fp32 accumulator from `ct.full`, then
   divide, row-sum and scale.
9. `gemm_tensor_core`: `A @ B + bias[N]`, fp16/bf16 operands, fp32 accumulation and
   output. This task has two extra suite gates (100% authored provenance and a
   performance requirement; see `perf_required` in `utils/task_suite.py`). Do not
   weaken either. If the cuTile solution cannot meet them, report the measured
   numbers and stop.

**Stretch (only after Tiers 1-3 pass on a GPU, and only with the user's go-ahead)**

10. `lora_shrink_bgmv`, then `flash_attention_fwd`.

**Out of scope, with the reason**

- `layernorm_residual_gelu`: its contract is exact erf GELU and cuTile 1.8.0 has no
  `erf`. Do not substitute a tanh or polynomial approximation.
- `topk_sampling`, `eagle_candidate_topk`: cuTile 1.8.0 has no sort or top-k.
- `prefix_sum`: the contract is one global scan; `ct.cumsum` scans within a tile.
  It is also `eval_only`.
- `grouped_gemm` (Triton-only, list inputs), `fp8_gemm` and `int8_gemm` (contract only).

If you find another task is not expressible, say which operation is missing and
skip it. Never change a contract to fit the backend.

## Layout per task

```
tasks/<task>/kernels/cutile/__init__.py     # lazy __getattr__ export of <task>_forward
tasks/<task>/kernels/cutile/<task>.py       # @ct.kernel functions + thin launch wrapper
tasks/<task>/model_new_cutile.py            # BACKEND = "cutile"; forward calls the wrapper
```

- `ModelNew` matches the reference constructor and `state_dict`.
- Wrappers may allocate (`torch.empty`, `torch.empty_like`), take views and read
  metadata. No Torch compute, tensor arithmetic, reductions or functional calls.
- Launch with `ct.launch(torch.cuda.current_stream(), grid, kernel, args)`. All
  device work joins the current stream before `forward` returns.
- Import `cuda.tile` only inside the kernel module, never from a Torch-free parent
  module (`utils/config.py`, `reward.py`, `grpo.py`, `rollout_env.py`, the catalog).
- No host syncs (`.item()`, `.tolist()`, `.nonzero()`) and no data-dependent shapes
  in `forward`.
- Regenerate each task `README.md` with `scripts/task_readme.py` rather than editing
  it by hand.

## cuTile rules that decide correctness here

- **Tails.** `ct.load`/`ct.store` move whole tiles. Catalog shapes are mostly powers
  of two, but 100,000-element and 128,256-wide cases are not. Use
  `ct.gather`/`ct.scatter` (out-of-bounds reads give `padding_value`, out-of-bounds
  writes are skipped) or an explicit tail. For reductions, a zero-filled tail is
  only correct for sums; `max` needs `padding_value` set to the identity.
- **Accumulate in fp32**, then cast with `.astype(out.dtype)`. This matches the
  reference contracts ("fp32 accumulation", "output input dtype").
- **Tile sizes are `ct.Constant[int]` arguments.** Each distinct value compiles a
  separate kernel. Derive them from the case shape only, keep the set small, and
  never derive them from tensor contents.
- **Kernel names** are how the profiler credits device time: the cuTile pattern is
  `^{name}_Kt\d+_`. Names need at least four characters and must not be one of the
  rejected names in SKILL_CUTILE.md (`add`, `sum`, `fused`, `kernel`, `mul`, ...).
  Use names such as `rmsnorm_cutile_rows`.
- **No library time.** `library_allowed=False` for cuTile. If `ct.mma` or `ct.matmul`
  shows up in the profile under a cuBLAS/CUTLASS symbol rather than your kernel's
  name, that row fails provenance. Record the observed symbols; do not edit the
  provenance rules to make it pass.
- **Do not use `ct.tune` inside `forward`** or anywhere on the timed path. Fixed,
  documented tile shapes only. Tuning, if used at all, is offline and its result is
  a constant in the source.
- **`allow_tma`, `latency`, `mma_scaled`, `grid_dependency_control_*`, `atomic_*`**
  exist in 1.8.0 but have never been run in this project. Use them only if a
  measured H200 result justifies it, and record that result.

## What cuTile 1.8.0 provides

Read from the pinned `cuda-tile==1.8.0` wheel (`cuda/tile/__init__.py`, `__all__`).
Nothing outside this list may be assumed to exist.

| Group | Names |
|---|---|
| Entry points | `kernel`, `function`, `launch`, `compiler_timeout`, `ByTarget` |
| Types and annotations | `Array`, `Tile`, `TiledView`, `Scalar`, `ScalarInt64`, `IndexedWithInt64`, `Slice`, `Constant`, `ArrayAnnotation`, `ConstantAnnotation`, `ListAnnotation`, `DType` |
| Dtypes | `bool_`, `int8/16/32/64`, `uint8/16/32/64`, `float16`, `bfloat16`, `float32`, `float64`, `tfloat32`, `float8_e4m3fn`, `float8_e5m2`, `float8_e8m0fnu`, `float8_e5m3fnu`, `float4_e2m1fn`, `foreign_pointer_dtype` |
| Grid | `bid`, `num_blocks`, `num_tiles` |
| Memory | `load`, `store`, `gather`, `scatter`, `load_advanced_indexing`, `store_advanced_indexing`; `Array.tiled_view(...)` with `TiledView.load/store/atomic_store_*`; `Array.slice` |
| Atomics | `atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_or`, `atomic_xchg`, `atomic_xor`; `MemoryOrder`, `MemoryScope` |
| Creation | `arange`, `full`, `ones`, `zeros`, `astile` |
| Shape | `reshape`, `permute`, `transpose`, `expand_dims`, `broadcast_to`, `cat`, `extract`, `insert` |
| Casts | `astype`, `bitcast`, `pack_to_bytes`, `unpack_from_bytes`; `RoundingMode`, `PaddingMode` |
| Matrix multiply | `mma(x, y, acc)`, `mma_scaled`, `matmul` |
| Reductions | `sum`, `prod`, `max`, `min`, `argmax`, `argmin`, `reduce` |
| Scans | `cumsum`, `cumprod`, `scan` |
| Arithmetic | `add`, `sub`, `mul`, `truediv`, `floordiv`, `cdiv`, `mod`, `divmod`, `pow`, `negative`, `abs`, `minimum`, `maximum`, `wrapping_add`, `wrapping_sub`, `wrapping_mul` |
| Math | `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `atan2`, `floor`, `ceil`, `isnan` |
| Comparison and select | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `where` |
| Bitwise | `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `bitwise_lshift`, `bitwise_rshift` |
| Static and debug | `static_eval`, `static_assert`, `static_iter`, `static_exception`, `ensure_constant`, `assume_divisible_by`, `assert_`, `print`, `printf` |
| Grid dependency | `grid_dependency_control_wait`, `grid_dependency_control_launch_dependents` |
| Exceptions | `TileError` and its subclasses (`TileSyntaxError`, `TileTypeError`, `TileValueError`, `TileUnsupportedFeatureError`, `TileCompilerExecutionError`, `TileCompilerTimeoutError`, ...) |

Submodules shipped in the wheel:

- `cuda.tile.tune`: `exhaustive_search`, `TuningResult`, `Measurement`.
- `cuda.tile.compilation`: `export_kernel`, `mangle_kernel_name`,
  `demangle_kernel_name`, `KernelSignature`, `CallingConvention` and the
  `*Constraint` classes.
- `cuda.tile.jax`: `cutile_call`, `OutputPlaceholder`, `InputOutput`. JAX is not a
  dependency of this project; do not use it.
- `cutile_cache` and the `cutile-cache` command: inspect the compilation cache.

Not in 1.8.0: `erf`, `sigmoid`, `softmax`, `log1p`, `expm1`, `clip`, `round`, sort,
top-k, `any`/`all`, logical operators (use the bitwise ones on `bool_`).

Install facts: the wheel requires Python 3.11-3.15 and `typing-extensions` only. The
`[tileiras]` extra adds `cuda-toolkit[nvcc,nvvm,tileiras]>=13.2,<13.6`. The pin in
`requirements-cutile.txt` and the Modal image (cuda-tile 1.8.0, tileiras 13.3.36)
stay as they are. `CUDA_TILE_LOGS=TILEIR` and `CUDA_TILE_DUMP_TILEIR` need CTK 13.4+
and are unavailable under this pin; `CUDA_TILE_LOGS=CUTILEIR` works.

## Catalog and suite wiring

- Declare cuTile per task in `utils/task_catalog.py` as a **stretch backend**
  (`stretch_backends=("cutile",)`) until it has passed on an H200. Promoting it to
  `backends` changes the row counts of default suite runs and is the user's decision.
- First read how `run_suite` and `scaffold` treat `stretch_backends` and the
  `hack`/`wrong` fixtures for a third backend (follow `gemm_tensor_core` + `cutedsl`).
  If selecting `--backends cutile` needs a code change, make the smallest one, keep
  the default argv and default row sets unchanged, and add a test.
- Existing result keys keep their names and meanings. Add keys only if needed.

## Tests

`cuda-tile` is not installed in the local `.venv` and cuTile has no CPU interpreter,
so no cuTile kernel can execute on this host. Local tests are therefore source-level:

- Each `kernels/cutile/<task>.py` parses, and `utils.backend` detects the workdir as
  `cutile` and discovers the authored kernel names (extend `tests/test_dsl_kernels.py`
  and `tests/test_symbol_provenance.py` in their existing style).
- Kernel names pass the naming rules; no banned Torch compute appears in wrappers or
  `model_new_cutile.py`.
- Catalog tests cover the new `stretch_backends` entries; `scripts/task_readme.py`
  output matches the committed READMEs.
- Both suites stay green:
  `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover` and
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover`.

Label these as source checks. They are not evidence that a kernel compiles, is
correct or is fast.

## GPU validation (prepare only; do not launch)

Do not launch any Modal run. Paid GPU runs need the user's approval, and Claude
launches them. Prepare, per tier:

```bash
.venv/bin/python -m scripts.freeze_task_suite --tasks <comma-separated tasks> --output <frozen dir>
cd <frozen dir> && PYTHONPATH=$PWD MODAL_GPU=H200 <repo>/.venv/bin/modal run --detach cloud/modal_tasks.py \
  --tasks <tasks> --case-set smoke --backends cutile --fixtures all --timeout-s 1200 --run-id <id>
```

State the expected row count and an estimated wall time with its basis. On the GPU,
each honest row must show: verification passed, authored provenance >= 0.9 (1.0 for
`gemm_tensor_core`), no library time, and the observed kernel symbols recorded, since
the `_Kt\d+_` naming is undocumented and was only observed for AXPBY.

## Rules that do not bend

- Never fabricate numbers. Until a GPU run exists, every cuTile timing, speedup and
  provenance share is "not measured", with the reason.
- No new Python dependencies. `cuda-tile` stays optional and GPU-only.
- Protected files stay byte-identical (`artifacts/upstream.json`).
- No changes to reward semantics, timing meanings, verification tolerances or
  provenance thresholds. Ask the user first if one seems necessary.
- `.git` is read-only in WSL: do not commit. Claude reviews, tests and commits from
  the Windows checkout.

## Report back

Write `HANDOFF_CODEX_CUTILE.md` (gitignored) with: the tasks completed and skipped
(with the missing operation for each skip), files added per task, the tile-shape and
grid choice per kernel and why, both local test counts, the prepared Modal commands
with row counts, and every open question for the user.
