# TileLang kernel optimisation

Accelerate model.py using kernels you author in TileLang. Aim for device time
strictly below 0.95 * torch.compile time while preserving correctness
(atol/rtol=1e-2).

## Layout and restrictions

- model.py, binding.cpp, binding_registry.h and utils/ are fixed policy infrastructure.
- Write kernels in kernels/tilelang/*.py as `@T.prim_func` functions (with
  `import tilelang.language as T`), or as `@tilelang.jit` kernel functions.
  Export thin wrappers from kernels/tilelang/__init__.py.
- Copy model_new_tilelang.py to model_new.py to start. ModelNew must match the
  reference constructor and state_dict; import kernels.tilelang or set BACKEND="tilelang".
- Compile with the default CUDA target (`target="cuda"`). The experimental
  `cutedsl`, `llvm` and other targets are not supported here.
- No torch.* compute operations, tensor arithmetic, @, reductions, or functional
  calls in model_new.py or wrappers. Allocation (torch.empty_like, TileLang
  outputs), views and metadata (shape, stride, dtype, device, numel) are allowed.
  Compute belongs in TileLang kernels.
- No third-party kernels or prewritten compute libraries. Use T.gemm for GEMM.
  cuBLAS/cuDNN/CUTLASS time is not allowed library time on TileLang.
- Give every kernel a distinctive name: TileLang names the device symbol after
  the prim_func (`<name>_kernel`), and the trusted profiler credits device time
  by that name. Names must reflect the kernels you actually authored.
- ModelNew.forward must enqueue all device work on torch.cuda.current_stream()
  of the inputs' device, or make that stream wait on side streams before
  returning; un-joined side-stream work under-reports event latency.
- No web tools during the policy rollout, harness edits, profiler tampering,
  kernel-name spoofing, importing the reference forward, or cached answers.

## Workflow

1. Profile the reference and inspect the task shapes, dtypes and memory layout.
2. Implement fused kernels and thin wrappers; compile once:
   `python -m utils.compile --backend tilelang` (or `bash utils/compile.sh --backend tilelang`).
3. Run `python -m utils.verification`: five independent random inputs, 1e-2 tolerances.
4. Run `python -m utils.profiling --backend tilelang --json --iters 10`.
   The generated path gets 20 warmups; any autotuning must finish before timing.
   For repeated event/wall statistics outside the profiler, run
   `python -m utils.benchmark --backend tilelang --json`; its bench/* keys are diagnostics only.
5. Iterate on the bottleneck, then evaluate the complete task:
   `python -m utils.rollout_env . --backend tilelang --max-turns 20 --turns-used 1`.
   Read the runtime RolloutConfig: your turn budget is **max_turns** (default 20).
   Exceeding it zeroes eligibility for positive reward: the numeric reward is -1,
   reason max_turns. DAPO and CISPO may subtract up to 1 within their buffer below that cap.
   The agent loop must enforce max_context_tokens; the environment enforces timeout_s.

## Optimisation checklist

- Fuse reads/writes and eliminate intermediates before tuning launch parameters.
- Stage tiles explicitly: T.copy global -> T.alloc_shared -> T.alloc_fragment,
  and back; keep GEMM accumulators in a float32 fragment.
- Pipeline K loops with T.Pipelined(..., num_stages=...); sweep block sizes and
  `threads` in T.Kernel; balance occupancy against shared-memory and register use.
- TileLang inserts tail bounds checks for shapes that are not tile multiples;
  they can block vectorisation, so prefer tile-aligned paths where shapes allow.
- Specialise on static shapes, or use T.dynamic(...) when one kernel must
  serve many sizes; never key compiled variants on random data.
- Inspect the generated CUDA with `kernel.get_kernel_source()`;
  TILELANG_PRINT_ON_COMPILATION=1 prints each kernel name as it compiles.
- There is no CPU interpreter: correctness and timing need a CUDA GPU.

## Reward and provenance

At least provenance_threshold (default 90%) of non-memcpy/memset device time
must belong to your source-discovered TileLang kernels. Unknown kernels and
torch/aten compute count against this share. Name-based attribution is checked
by the trusted profiler; names must reflect actual authored kernels. CPU or
unmeasured runs cannot earn a performance reward. Correctness/provenance failure
returns -1; otherwise the base reward is 3 for >5% faster than both eager and
compile, 2 for eager only, and 1 otherwise. GRPO keeps tied groups; DAPO and CISPO filter
them by default. Always use the supplied RolloutConfig: these switches and
thresholds may be overridden.

## Kernel naming and attribution

Authored names must have at least four characters and must not be any of:
`add`, `aten`, `copy`, `div`, `elementwise`, `fill`, `forward`, `fused`, `kernel`, `main`, `mul`, `native`, `sub`, `sum`, `triton`, `vectorized`. Rejected names are excluded
from source discovery and cannot receive agent credit.

The profiler uses `re.search` with the backend pattern `^{name}(_kernel)?$`,
substituting `re.escape(authored_name)` for `{name}`. Patterns are based on
H200 run `20261106T185236Z`. Memory events are excluded first; Torch namespace
markers (`at::native`, `aten::`, `c10::`) and Inductor names matching
`^triton_(poi|per|red|tem)_fused` always count as Torch before authored matching.
Unknown events count as Torch. CuTe symbols containing `cutlass` are checked
for authored matches before library markers. Matching remains name-based,
not proof of authorship.
