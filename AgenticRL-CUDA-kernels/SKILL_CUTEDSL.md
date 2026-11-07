# CuTe DSL kernel optimisation

Accelerate model.py using kernels you author in CuTe DSL (CUTLASS 4.x,
`import cutlass.cute as cute`). Aim for device time strictly below
0.95 * torch.compile time while preserving correctness (atol/rtol=1e-2).

## Layout and restrictions

- model.py, binding.cpp, binding_registry.h and utils/ are fixed policy infrastructure.
- Write device kernels as `@cute.kernel` functions and their host launchers as
  `@cute.jit` functions in kernels/cutedsl/*.py; export thin wrappers from
  kernels/cutedsl/__init__.py. Only `@cute.kernel` functions are credited as
  authored kernels.
- Copy model_new_cutedsl.py to model_new.py to start. ModelNew must match the
  reference constructor and state_dict; import kernels.cutedsl or set BACKEND="cutedsl".
- Pass torch tensors through `cutlass.cute.runtime.from_dlpack`. Compile each
  launcher once with `cute.compile` and reuse the compiled function; never
  compile inside a timed call.
- No torch.* compute operations, tensor arithmetic, @, reductions, or functional
  calls in model_new.py or wrappers. Allocation (torch.empty_like), views and
  metadata (shape, stride, dtype, device, numel) are allowed. Compute belongs in
  CuTe DSL kernels.
- No prewritten CUTLASS kernels, cutlass.utils GEMM/convolution templates,
  third-party kernels or other compute libraries. Build GEMMs from CuTe MMA and
  copy atoms yourself. cuBLAS/cuDNN/CUTLASS-library time is not allowed library
  time on CuTe DSL.
- CuTe DSL names device symbols after the kernel function
  (`kernel_cutlass_<name>_..._0` by default) and the trusted profiler credits
  device time by that name. Keep the default naming (no `remove_cutlass_symbol`),
  and give kernels names that reflect what you actually authored.
- ModelNew.forward must enqueue all device work on torch.cuda.current_stream()
  of the inputs' device: pass `cutlass.torch.current_stream()` to your launcher
  and launch with `stream=...`. Un-joined side-stream work under-reports event latency.
- No web tools during the policy rollout, harness edits, profiler tampering,
  kernel-name spoofing, importing the reference forward, or cached answers.

## Workflow

1. Profile the reference and inspect the task shapes, dtypes and memory layout.
2. Implement fused kernels and thin wrappers; compile once:
   `python -m utils.compile --backend cutedsl` (or `bash utils/compile.sh --backend cutedsl`).
3. Run `python -m utils.verification`: five independent random inputs, 1e-2 tolerances.
4. Run `python -m utils.profiling --backend cutedsl --json --iters 10`.
   The generated path gets 20 warmups; compilation must finish before timing.
   For repeated event/wall statistics outside the profiler, run
   `python -m utils.benchmark --backend cutedsl --json`; its bench/* keys are diagnostics only.
5. Iterate on the bottleneck, then evaluate the complete task:
   `python -m utils.rollout_env . --backend cutedsl --max-turns 20 --turns-used 1`.
   Read the runtime RolloutConfig: your turn budget is **max_turns** (default 20).
   Exceeding it zeroes eligibility for positive reward: the numeric reward is -1,
   reason max_turns. DAPO and CISPO may subtract up to 1 within their buffer below that cap.
   The agent loop must enforce max_context_tokens; the environment enforces timeout_s.

## Optimisation checklist

- Fuse reads/writes and eliminate intermediates before tuning layouts.
- Choose thread and value layouts (`make_layout_tv`) so each thread moves
  128-bit vectors; predicate tails with an identity tensor and `elem_less`.
- Mark tensors `mark_layout_dynamic(...)` when one compiled kernel must serve
  many shapes; static layouts allow more specialisation but compile per shape.
- For GEMM, stage tiles through shared memory with copy atoms (cp.async/TMA
  where the architecture supports it) and use MMA atoms with float32 accumulators.
- Set CUTE_DSL_LINEINFO=1 to correlate profiler output with Python source.
- There is no CPU interpreter: correctness and timing need a CUDA GPU.

## Reward and provenance

At least provenance_threshold (default 90%) of non-memcpy/memset device time
must belong to your source-discovered `@cute.kernel` functions. Unknown kernels
and torch/aten compute count against this share. Name-based attribution is
checked by the trusted profiler; names must reflect actual authored kernels.
CPU or unmeasured runs cannot earn a performance reward. Correctness/provenance
failure returns -1; otherwise the base reward is 3 for >5% faster than both eager
and compile, 2 for eager only, and 1 otherwise. GRPO keeps tied groups; DAPO
and CISPO filter them by default. Always use the supplied RolloutConfig: these switches
and thresholds may be overridden.

## Kernel naming and attribution

Authored names must have at least four characters and must not be any of:
`add`, `aten`, `copy`, `div`, `elementwise`, `fill`, `forward`, `fused`, `kernel`, `main`, `mul`, `native`, `sub`, `sum`, `triton`, `vectorized`. Rejected names are excluded
from source discovery and cannot receive agent credit.

The profiler uses `re.search` with the backend pattern `^kernel_cutlass_{name}_`,
substituting `re.escape(authored_name)` for `{name}`. Patterns are based on
H200 run `20261106T185236Z`. Memory events are excluded first; Torch namespace
markers (`at::native`, `aten::`, `c10::`) and Inductor names matching
`^triton_(poi|per|red|tem)_fused` always count as Torch before authored matching.
Unknown events count as Torch. CuTe symbols containing `cutlass` are checked
for authored matches before library markers. Matching remains name-based,
not proof of authorship.
