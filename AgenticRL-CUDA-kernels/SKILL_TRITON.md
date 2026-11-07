# Triton kernel optimisation

Accelerate model.py using kernels you author. Aim for device time strictly
below 0.95 * torch.compile time while preserving correctness (atol/rtol=1e-2).

## Layout and restrictions

- model.py, binding.cpp, binding_registry.h and utils/ are fixed policy infrastructure.
- Write @triton.jit kernels in kernels/triton/*.py; export wrappers from __init__.py.
- Copy model_new_triton.py to model_new.py to start. ModelNew must match the
  reference constructor and state_dict; import kernels.triton or set BACKEND="triton".
- No torch.* compute operations, tensor arithmetic, @, reductions, or functional
  calls in model_new.py or wrappers. Allocation (torch.empty_like) and metadata
  (shape, stride, dtype, device, numel) are allowed. Compute belongs in jit kernels.
- No triton.ops, flash-attn, third-party kernels, or prewritten compute libraries.
  Use tl.dot for GEMM. cuBLAS/cuDNN/CUTLASS time is not allowed library time on Triton.
- No web tools during the policy rollout, harness edits, profiler tampering,
  kernel-name spoofing, importing the reference forward, or cached answers.
- ModelNew.forward must enqueue all device work on torch.cuda.current_stream()
  of the inputs' device, or make that stream wait on side streams before
  returning; un-joined side-stream work under-reports event latency.

## Workflow

1. Profile the reference and inspect the task shapes, dtypes and memory layout.
2. Implement fused kernels and thin wrappers; compile/JIT/autotune once:
   `python -m utils.compile --backend triton` (or `bash utils/compile.sh --backend triton`).
3. Run `python -m utils.verification`: five independent random inputs, 1e-2 tolerances.
4. Run `python -m utils.profiling --backend triton --json --iters 10`.
   The generated path gets 20 warmups; autotuning must finish before timing.
   For repeated event/wall statistics outside the profiler, run
   `python -m utils.benchmark --json`; its bench/* keys are diagnostics only.
5. Iterate on the bottleneck, then evaluate the complete task:
   `python -m utils.rollout_env . --backend triton --max-turns 20 --turns-used 1`.
   Read the runtime RolloutConfig: your turn budget is **max_turns** (default 20).
   Exceeding it zeroes eligibility for positive reward: the numeric reward is -1,
   reason max_turns. DAPO may subtract up to 1 within its buffer below that cap.
   The agent loop must enforce max_context_tokens; the environment enforces timeout_s.

## Optimisation checklist

- Fuse reads/writes and eliminate intermediates before tuning launch parameters.
- Sweep BLOCK sizes, num_warps, num_stages; balance occupancy and register pressure.
- Mask tail loads/stores, including short and non-divisible shapes; test empty inputs.
- Coalesce vectorised loads and stores; respect the actual strides and alignment.
- Choose autotune keys from dimensions that affect performance; never key on random data.
- For tl.dot, check accumulation precision, tiling, layout and shared-memory pressure.
- Debug with `CUDA_AGENT_DEVICE=cpu TRITON_INTERPRET=1 python -m utils.verification`.
  The interpreter uses NumPy and pins one config to avoid GPU autotune timers;
  CPU correctness does not establish GPU timing or provenance.

## Reward and provenance

At least provenance_threshold (default 90%) of non-memcpy/memset device time
must belong to your source-discovered jit kernels. Unknown kernels and torch/aten
compute count against this share. Name-based attribution is checked by the trusted
profiler; names must reflect actual authored kernels. CPU/unmeasured runs cannot
earn a performance reward. Correctness/provenance failure returns -1; otherwise
the base reward is 3 for >5% faster than both eager and compile, 2 for eager only,
and 1 otherwise. GRPO keeps tied groups; DAPO filters them by default. Always use
the supplied RolloutConfig: these switches and thresholds may be overridden.

## Kernel naming and attribution

Authored names must have at least four characters and must not be any of:
`add`, `aten`, `copy`, `div`, `elementwise`, `fill`, `forward`, `fused`, `kernel`, `main`, `mul`, `native`, `sub`, `sum`, `triton`, `vectorized`. Rejected names are excluded
from source discovery and cannot receive agent credit.

The profiler uses `re.search` with the backend pattern `^{name}(_\d\w*)?$`,
substituting `re.escape(authored_name)` for `{name}`. Patterns are based on
H200 run `20261106T185236Z`. Memory events are excluded first; Torch namespace
markers (`at::native`, `aten::`, `c10::`) and Inductor names matching
`^triton_(poi|per|red|tem)_fused` always count as Torch before authored matching.
Unknown events count as Torch. CuTe symbols containing `cutlass` are checked
for authored matches before library markers. Matching remains name-based,
not proof of authorship.
