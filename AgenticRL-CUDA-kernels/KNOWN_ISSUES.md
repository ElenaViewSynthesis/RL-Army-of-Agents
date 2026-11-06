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
- **Benchmark mode unmeasured on this host.** Only the CPU path is exercised
  here: `bench/available: false` plus statistics/rotation fixtures. Every real
  `bench/*` latency, device key and CUDA-graph result needs a GPU run. A
  benchmark stage that crashes, times out or emits malformed output only
  yields `bench/available: false` with a reason; it never changes the reward.
- **Event latency excludes un-joined side streams.** `bench/*/event_us_*`
  covers only work ordered on the entry stream; wall latency brackets
  `torch.cuda.synchronize()` and includes CPU dispatch and every stream.
  Candidates launching side streams must join the current stream before
  returning, or `bench/generated/event_wall_ratio` exposes the gap.
- **CUDA-graph capture is a diagnostic, not a score.** It requires
  current-stream launches and no host syncs (`.item()`, `.cpu()`,
  `torch.cuda.synchronize()`) or value-dependent control flow in forward.
  Failures set `bench/*/graph_capturable: false` plus `graph_error` and never
  change the reward.
- **Hopper needs tileiras 13.3+.** The cuTile PyPI README states a lower floor;
  sm_90 support arrived with tileiras/CTK 13.3 (cuTile 1.4.0 notes), so the
  pin is `>=13.3,<13.4`. Modal's driver API is CUDA 13.3, which also bounds the
  container toolkit version.
- **pip-installed nvcc is not a CUDA_HOME.** The CUDA backend relies on the
  devel image's `/usr/local/cuda` headers and toolchain; wheels alone are not
  sufficient for `torch.utils.cpp_extension`.
- **DSL backends are unvalidated on hardware.** The TileLang, cuTile and CuTe DSL
  kernels have only CPU-side discovery and classification tests here; their GPU
  kernel and run_task tests skip without a CUDA GPU and the package. None of
  these DSLs ships a CPU interpreter in its release packages.
- **DSL device symbol names are partly undocumented.** TileLang names symbols
  `<prim_func>_kernel` (its docs show `main` -> `main_kernel`); CuTe DSL defaults
  to `kernel_cutlass_<fn>_<args>_0` (CUTLASS 4.7 docs); cuTile does not document
  its naming. Attribution assumes each symbol contains the authored function
  name. Record real `ctx.events()` names on the GPU host before trusting DSL
  provenance; if a name doesn't match, report it rather than loosening
  unknown-event handling.
- **Generic kernel names over-credit provenance (all backends).** The classifier
  credits any device event whose name contains an authored kernel name as a
  substring. A kernel named something generic (`kernel`, `add`, or TileLang's
  conventional `main`) also matches unrelated PyTorch kernels such as
  `at::native::vectorized_elementwise_kernel` or `remainder_kernel`, so a
  Torch-op substitution plus a dummy kernel could pass the 90% gate. This
  predates the DSL backends. Fixing it needs a decision, because either option
  changes current CUDA/Triton behaviour: anchor matches to each backend's
  symbol pattern (e.g. CUDA `name<`/`name(`, TileLang `name_kernel`, CuTe DSL
  `kernel_cutlass_name_`), or refuse to credit generic and very short names.
- **Optional DSL packages may conflict.** TileLang wheels bundle their own TVM,
  `nvidia-cutlass-dsl` pulls CUDA 12 runtime libraries by default, and
  `cuda-tile[tileiras]` pulls nvcc/nvvm 13.x. Install each extra in its own
  environment if they clash with each other or with the torch CUDA build.
- **Only TileLang gets a task-local compile cache.** `TILELANG_CACHE_DIR` is set
  per task like `TRITON_CACHE_DIR`. No cache location is configured for cuTile
  or CuTe DSL; if they don't cache to disk, each compile, verification and
  profiling subprocess recompiles, and that counts against the task deadline.
- **Deferred scope.** No actual training, Modal orchestration, policy weights,
  agent loop, model registry or observability services are added. The optional
  Level-2 GEMM stretch kernel is deferred to prioritise the required pipeline
  and correctness/timeout tests. Context length is a shared config field for
  the future loop, not something the task evaluator can observe or enforce.
- **Controlled comparison.** The requested presets also set GRPO kl_coef=0.001
  and DAPO kl_coef=0.0. For a comparison differing only in the four specified
  toggles, explicitly hold `--kl-coef` equal, along with seeds and budgets.
