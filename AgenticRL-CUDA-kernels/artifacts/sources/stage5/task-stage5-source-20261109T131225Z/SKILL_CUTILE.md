# cuTile kernel optimisation

Accelerate model.py using kernels you author in cuTile (NVIDIA CUDA Tile,
`import cuda.tile as ct`). Aim for device time strictly below
0.95 * torch.compile time while preserving correctness (atol/rtol=1e-2).

## Layout and restrictions

- model.py, binding.cpp, binding_registry.h and utils/ are fixed policy infrastructure.
- Write `@ct.kernel` functions in kernels/cutile/*.py; export thin wrappers from
  kernels/cutile/__init__.py. Kernels cannot be called from host code: launch them
  with `ct.launch(torch.cuda.current_stream(), grid, kernel, args)`.
- Copy model_new_cutile.py to model_new.py to start. ModelNew must match the
  reference constructor and state_dict; import kernels.cutile or set BACKEND="cutile".
- No torch.* compute operations, tensor arithmetic, @, reductions, or functional
  calls in model_new.py or wrappers. Allocation (torch.empty_like), views and
  metadata (shape, stride, dtype, device, numel) are allowed. Compute belongs in
  cuTile kernels.
- No third-party kernels or prewritten compute libraries. Use `ct.mma` (with a
  float32 accumulator from `ct.full`) for GEMM. cuBLAS/cuDNN/CUTLASS time is not
  allowed library time on cuTile.
- The trusted profiler credits device time by the names of your `@ct.kernel`
  functions; names must reflect the kernels you actually authored.
- ModelNew.forward must enqueue all device work on torch.cuda.current_stream()
  of the inputs' device, or make that stream wait on side streams before
  returning; un-joined side-stream work under-reports event latency.
- No web tools during the policy rollout, harness edits, profiler tampering,
  kernel-name spoofing, importing the reference forward, or cached answers.

## Workflow

1. Profile the reference and inspect the task shapes, dtypes and memory layout.
2. Implement fused kernels and thin wrappers; compile once (first launch):
   `python -m utils.compile --backend cutile` (or `bash utils/compile.sh --backend cutile`).
3. Run `python -m utils.verification`: five independent random inputs, 1e-2 tolerances.
4. Run `python -m utils.profiling --backend cutile --json --iters 10`.
   The generated path gets 20 warmups; compilation and tuning must finish before timing.
   For repeated event/wall statistics outside the profiler, run
   `python -m utils.benchmark --backend cutile --json`; its bench/* keys are diagnostics only.
5. Iterate on the bottleneck, then evaluate the complete task:
   `python -m utils.rollout_env . --backend cutile --max-turns 20 --turns-used 1`.
   Read the runtime RolloutConfig: your turn budget is **max_turns** (default 20).
   Exceeding it zeroes eligibility for positive reward: the numeric reward is -1,
   reason max_turns. DAPO and CISPO may subtract up to 1 within their buffer below that cap.
   The agent loop must enforce max_context_tokens; the environment enforces timeout_s.

## Optimisation checklist

- Fuse reads/writes and eliminate intermediates before tuning tile shapes.
- Pass tile sizes as `ct.Constant[int]` parameters; each distinct value compiles
  a separate kernel, so keep the set small and never derive it from random data.
- `ct.load`/`ct.store` move whole tiles and assume in-bounds tiles; for shapes
  that are not tile multiples use `ct.gather`/`ct.scatter` (zero-fill and skip
  out-of-bounds elements) or handle the tail explicitly.
- Accumulate reductions and matmuls in float32, then cast with `.astype(...)`
  to the output dtype.
- Sweep tile shapes and the grid; balance occupancy against register pressure.
- There is no CPU interpreter: correctness and timing need a CUDA GPU.

## Reward and provenance

At least provenance_threshold (default 90%) of non-memcpy/memset device time
must belong to your source-discovered cuTile kernels. Unknown kernels and
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

The profiler uses `re.search` with the backend pattern `^{name}_Kt\d+_`,
substituting `re.escape(authored_name)` for `{name}`. Patterns are based on
H200 run `20261106T185236Z`. Memory events are excluded first; Torch namespace
markers (`at::native`, `aten::`, `c10::`) and Inductor names matching
`^triton_(poi|per|red|tem)_fused` always count as Torch before authored matching.
Unknown events count as Torch. CuTe symbols containing `cutlass` are checked
for authored matches before library markers. Matching remains name-based,
not proof of authorship.
