# Known issues and remaining validation

- **No GPU or nvcc on this host.** CUDA compilation, real profiler event names,
  speedups, GPU provenance and GPU end-to-end reward comparisons cannot be
  measured here. Those integration tests skip explicitly. Unknown names count
  against provenance; no blanket `triton_` fallback is used because Torch
  Inductor generates that prefix too. Inspect real `ctx.events()` names on the
  target GPU before accepting the GPU validation as complete.
- **Interpreter autotune.** CPU interpretation pins the first block size; GPU
  execution still autotunes three block sizes. The interpreter uses NumPy,
  which is needed for CPU validation but is not an added GPU runtime dependency.
  bfloat16 interpreter support varies; tests cover float16/32/64.
- **Attribution limits.** Matching source-discovered names closes the specified
  ordinary Torch-op substitution, but is not proof of authorship against name
  spoofing, dynamically generated names, or mixed authored/helper launches.
  CUDA libraries without cublas/cutlass/cudnn in their event names fail closed.
  Conversely, CUDA's allowed-library bucket cannot distinguish a Torch-issued
  cuBLAS/cuDNN call from an agent-issued one using device names alone.
  Do not relax unknown-event handling to make a benchmark pass.
- **Process isolation is not a hostile-code sandbox.** POSIX process groups
  isolate crashes and terminate ordinary descendants. Phase 2 must supply the
  container, read-only trusted utilities, filesystem/network/resource limits,
  and a container-level timeout for GPU driver hangs or escaped processes.
  Agent policy skills forbid editing trusted code; local development files are
  not permission-locked by this implementation.
- **Modal cold starts and contention.** The 180-second deadline includes Python
  imports, nvcc/JIT/autotune, five verification trials, and torch.compile warmup.
  Cold Level-3 tasks may need a higher `--timeout-s`. Use prebuilt dependencies,
  and avoid concurrent profiling on one sandbox GPU: contention changes both
  milestone rewards and the group-relative training signal. Separate workdirs
  avoid build collisions; inherited cache overrides are the caller's responsibility.
  On this Windows-mounted host, the first CPU smoke compile took 143.19 seconds
  and the default task deadline expired in verification (raw command/result in
  artifacts/rollouts_cold_start.log). Local CPU validation uses `--timeout-s 600`;
  the production default is unchanged. The CUDA attempt also reports missing
  Ninja; a GPU image needs the upstream build tools (Ninja and nvcc).
- **Deferred scope.** No actual training, Modal orchestration, policy weights,
  agent loop, model registry or observability services are added. The optional
  Level-2 GEMM stretch kernel is deferred to prioritise the required pipeline
  and correctness/timeout tests. Context length is a shared config field for
  the future loop, not something the task evaluator can observe or enforce.
- **Controlled comparison.** The requested presets also set GRPO kl_coef=0.001
  and DAPO kl_coef=0.0. For a comparison differing only in the four specified
  toggles, explicitly hold `--kl-coef` equal, along with seeds and budgets.
