# Known issues and remaining validation

- **First GPU run: CUDA C++ standard mismatch.** Run `20261106T142830Z`
  reached the H200 and validated the Python backends, but stock CUDA
  compilation failed with `C++20 or later compatible compiler is required
  to use ATen.` The harness forced `-std=c++17`, overriding Torch's default.
  The override is removed; Torch now selects its required standard. The stock
  pipeline passed compilation and all five checks in run `20261106T185236Z`.
- **First Modal image build: dependency conflict.** The preflight attempt
  (`.venv/bin/modal run cloud/modal_runtime.py`, app
  `ap-vUlPvIT9iLLPF60u2pKDfc`, image `im-u7ElhT76HK3e4MiGfUc5JZ`)
  failed before GPU allocation: Torch 2.14.1+cu130 requires the
  `cuda-toolkit` runtime metapackage at 13.0.3, conflicting with the requested
  `cuda-toolkit[nvcc,nvvm,tileiras]>=13.3,<13.4`. The CUDA 13.3.1 devel
  base image built successfully. `requirements-gpu.txt` now pins the
  individual 13.3.1 compiler components (tileiras 13.3.36, nvcc/nvvm/CRT
  13.3.73, nvJitLink 13.3.33), preserving Torch's runtime dependencies.
  The rebuild succeeded and preflight `20261106T142735Z` passed on H200.
- **Local host remains CPU-only; first GPU evidence now exists.** Modal H200
  runs `20261106T142830Z` and `20261106T185236Z` captured real profiler
  names. The latter stock CUDA pipeline compiles and verifies after removing
  the C++17 override. See VALIDATION.md for measured results and raw logs. Unknown names count
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
- **Benchmark latency series were not archived.** H200 run
  `20261106T185236Z` passed GPU benchmark sampling and CUDA-graph tests
  (`Ran 30 tests in 380.509s`, `OK` for the combined suite). The tests
  assert samples and replay behavior but do not export raw latency series. A
  benchmark stage that crashes, times out or emits malformed output only
  yields `bench/available: false` with a reason; it never changes the reward.
- **Zero eager baseline in fresh H200 validation.** Run `20261108T014026Z`
  passed the 31-test GPU suite, but the separate honest CuTe DSL validation
  profile recorded `t_eager_us=0.0` despite `timing_measured=true`.
  Correctness and provenance passed; reward correctly remained -1 with reason
  `timing`. Command exit success does not make this full timing acceptance.
  The saved profile contains generated-kernel events, not the eager baseline's
  event list or a trace, so the cause remains unconfirmed. Preserve this failed
  measurement; collect baseline event evidence before changing profiler
  semantics or relying on sweep rewards. See VALIDATION.md and
  `artifacts/acceptance-review-20261108T014026Z.json`.
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
- **DSL hardware validation is limited to the shipped fixtures.** TileLang,
  cuTile and CuTe DSL executed on Modal H200 in run `20261106T142830Z`: honest
  fixtures verified with provenance 1.0; Torch-op hacks verified with provenance
  0.0. This establishes no performance claim for larger workloads or other GPUs.
- **Kernel names observed; naming remains a heuristic.** Run
  `20261106T185236Z` captured all five backend symbols (raw records in
  `artifacts/kernel_names.stdout-20261106T185236Z.log`). cuTile emitted
  `axpby_cutile_kernel_Kt1_A1f32_1i16t1_p16_A1f32_1i16t1_p16_A1f32_1i16t1_p16_Sf32_I128`;
  Triton emitted the bare `axpby_kernel`. TileLang and CuTe matched their
  documented forms. All had empty unmatched-name lists. Other shapes,
  types and compiler releases may differ; unknown events must still fail closed.
  For mismatches, see the [unmatched-name diagnostic runbook](VALIDATION.md#kernel-symbol-inspection-and-unmatched-name-diagnostics).
- **Generic-name over-credit fixed with anchored attribution.** Authored names
  must have at least four characters and avoid the shared generic-name denylist.
  Both discovery and classification enforce this. Memory is handled first;
  Torch/Inductor namespace markers precede backend-specific authored matching;
  library markers follow it. Rules and captured symbols from run
  `20261106T185236Z` are in VALIDATION.md. Unknown events still count as Torch.
  Run `20261106T231526Z` confirmed all honest fixtures at 100% and all five
  launched dummy-kernel spoofs at 0% (reward -1); the GPU suite passed 31 tests.
  Attribution remains name-based, not proof of authorship. CUDA's library
  policy still credits `cutlass` markers even for an unmatched CuTe symbol.
  The exact approved Triton and TileLang patterns overlap on bare names if
  given the same authored name; each rollout uses its own backend's discovery.
- **Optional DSL packages may conflict.** TileLang wheels bundle their own TVM,
  `nvidia-cutlass-dsl` pulls CUDA 12 runtime libraries by default, and
  `cuda-tile[tileiras]` pulls nvcc/nvvm 13.x. Install each extra in its own
  environment if they clash with each other or with the torch CUDA build. The
  cloud image installs all three together; the first build required the compiler
  pin fix above, after which all three executed successfully on H200.
- **Only TileLang gets a task-local compile cache.** `TILELANG_CACHE_DIR` is set
  per task like `TRITON_CACHE_DIR`. No cache location is configured for cuTile
  or CuTe DSL; if they don't cache to disk, each compile, verification and
  profiling subprocess recompiles, and that counts against the task deadline.
- **Tracking is opt-in and unverified against the live service.** Payloads
  were checked with fake SDKs, `--dry-run`, and a real `wandb==0.31.0` offline
  smoke using explicitly synthetic records. Real-SDK config, tables, summaries
  and staged evidence passed locally; this is not live W&B/Weave validation.
  The first approved online backfill must be inspected in the W&B UI.
  `${env:WANDB_API_KEY}`
  header expansion in `.devin/mcp_config.json` follows Devin's OAuth-field
  docs and should be confirmed on first use.
- **Deferred scope.** No actual training, Modal orchestration, policy weights,
  agent loop, model registry or observability services are added. The optional
  Level-2 GEMM stretch kernel is deferred to prioritise the required pipeline
  and correctness/timeout tests. Context length is a shared config field for
  the future loop, not something the task evaluator can observe or enforce.
- **Controlled comparison.** The requested presets also set GRPO kl_coef=0.001
  and DAPO and CISPO kl_coef=0.0. For a comparison differing only in the preset
  toggles, explicitly hold `--kl-coef` equal, along with seeds and budgets.

- **Stale cloud environment artifact corrected locally.** The runner originally
  copied the uploaded local CPU `artifacts/environment.json` into GPU results.
  Historical copies are preserved as `environment-uploaded-stale-<run_id>.json`.
  Corrected `environment-<run_id>.json` records are reconstructed from each
  run's fresh `require.stdout.log`, with source and command recorded. The runner
  now writes environment records from that probe. Run `20261106T231526Z`
  remotely verified the fix: every packaged runtime field matches the fresh
  require probe, and the run_id matches.
- **Commits unavailable in this workspace.** A focused commit attempt failed
  with `Unable to create .../.git/index.lock: Read-only file system`.
  Changes and evidence are retained in the working tree; nothing was pushed.
