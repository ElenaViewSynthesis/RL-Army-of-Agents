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
  Inductor generates that prefix too. Captured symbols are pinned in
  VALIDATION.md; the remaining timing blocker is the intermittent empty/zero
  eager baseline below, not unobserved symbols.
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
- **Historical benchmark latency series were not archived.** H200 run
  `20261106T185236Z` passed GPU benchmark sampling and CUDA-graph tests
  (`Ran 30 tests in 380.509s`, `OK` for the combined suite). The tests
  asserted samples and replay behavior without exporting raw latency series.
  The runner now sets `CUDA_AGENT_BENCH_SAMPLES_DIR` to archive per-sample
  series under the run volume's `benchmark-samples/`. A
  benchmark stage that crashes, times out or emits malformed output only
  yields `bench/available: false` with a reason; it never changes the reward.
- **Zero eager baseline in fresh H200 validation.** Run `20261108T014026Z`
  passed the 31-test GPU suite, but the separate honest CuTe DSL validation
  profile recorded `t_eager_us=0.0` despite `timing_measured=true`.
  Correctness and provenance passed; reward correctly remained -1 with reason
  `timing`. Command exit success does not make this full timing acceptance.
  The saved profile contains generated-kernel events, not the eager baseline's
  event list or a trace, so the cause remains unconfirmed. Preserve this failed
  measurement. Profiling now retains CUDA event counts for all three sessions
  and eager names; an empty session produces null time, `timing_measured=false`
  and `empty_profile:<session>` (reward -1), with no retry. `--trace-dir`
  preserves Chrome traces; traced/reordered captures are diagnostic and cannot
  supply rollout rewards. Diagnostic Run A (`20261108T091114Z` plus
  `20261108T103454Z`) captured 193 traced trials, 100 of them CuTe DSL, with no
  empty session: it did not reproduce the fault, and the cause remains
  unconfirmed. Untraced full runs `20261108T140200Z` and `20261108T153803Z`
  subsequently passed acceptance; the latter validates the approved conditioned
  method, not the earlier measurement method. Nonpositive/non-finite durations
  now also fail closed while retaining per-session samples. Neither pass proves
  the historical fault fixed. The explicitly requested fresh repeat
  `20261108T164430Z` also passed all five backends and the 31-test GPU suite,
  validating the unchanged conditioned scorer's nonempty-capture path.
  Its new evidence and source hashes are retained under
  `artifacts/fresh-gpu-20261108T164430Z/`. See VALIDATION.md and
  `artifacts/acceptance-review-20261108T014026Z.json`.
  Diagnosis runs `20261108T091114Z` (interrupted) and `20261108T103454Z`
  produced 193 complete trials, including 100 CuTe DSL trials split 50/50 by
  order, without an empty capture. The continuation saved 420 matching traces.
  This does not establish the cause of the original untraced failure.
- **Profiling order shifts the eager baseline by about 8%.** Run A measured eager
  at 1.73–1.77 µs (medians) when profiled first and 1.60–1.65 µs in the
  production generated-first order, across all five backends. At sub-2 µs sizes
  this can move `faster_than_eager` decisions near the margin. The cause is
  unexplained at the low level; the model order is unchanged. Controls
  `20261108T142102Z` show that one empty/eager primer reverses the effect and
  all-model warmup does not remove it. Extension controls `20261108T144950Z`
  reduce median eager order shifts to 0.15% (Triton) and 0.32% (CuTe DSL) with
  two intentional empty captures. With explicit approval, production now uses
  this conditioning before model timing, recording
  `cuda-leaf-sum-two-empty-prime-v1`; full run `20261108T153803Z` passes.
  This is a bounded mitigation, not a universal stability or CUPTI root-cause
  claim. Historical timings remain unchanged and are not equivalent measurements.
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
- **DSL cache directories are task-local.** Alongside `TRITON_CACHE_DIR` and
  `TILELANG_CACHE_DIR`, the harness now sets `CUDA_TILE_CACHE_DIR` (cuTile 1.8.0)
  and `CUTE_DSL_CACHE_DIR` (CuTe DSL 4.8.1). Pinned sources:
  [cuTile cache documentation](https://github.com/NVIDIA/cutile-python/blob/v1.8.0/docs/source/debugging.rst)
  and [CuTe cache implementation](https://github.com/NVIDIA/cutlass/blob/v4.8.1/python/CuTeDSL/cutlass/base_dsl/cache_helpers.py).
  Caller overrides still take precedence in rollouts. Diagnostic repetitions
  override them with fresh task directories. Disk caches do not establish that
  every subprocess avoids compilation; stage wall times include imports/JIT.
- **Native Windows is not supported by the process harness.** Six local test
  failures were reported on Windows (POSIX `os.killpg`, backslash paths and no
  `bash`); this Linux/WSL session has not reproduced them on native Windows.
  Use Linux/WSL2. These tests are retained to enforce process cleanup here.
- **Tracking remains opt-in; live SDK validation and approved backfill passed.**
  New run `tracking-check-df6b74c7f55f4490` passed W&B/Weave readback and user
  UI inspection. Its historical source remains partial, including CuTe DSL's
  zero baseline. Five approved manifest entries were then migrated: current
  summaries, latest tables/config flags, raw bytes and Weave outputs passed
  readback. Legacy metric history and missing historical metadata were preserved.
  Evidence: `artifacts/tracking-live-check-20261108T142009Z/report.json` and
  `artifacts/tracking-backfill-20261108T144457Z/report.json`.
  This validates SDK paths, not the separate MCP header expansion.
- **Deferred scope.** No actual training, optimizer, policy weights,
  agent loop or model registry is added. The optional
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

- **Historical Modal JWT expiry — unknown/non-blocking.** Diagnostic run
  `20261108T091114Z` (Modal SDK 1.6.3) ended with `AuthError: Jwt is expired`
  in `AttemptAwait`. The app stopped with 53 complete trial records (20 CUDA,
  20 Triton, 13 TileLang); none had an empty capture, and CuTe DSL had not run.
  Partial artifacts are preserved. The orchestration now uses one durable
  `spawn()` and repeated `get(timeout=30)` calls, without resubmitting the job;
  use `modal run --detach` for long diagnostics. This follows
  [Modal's invocation guidance](https://modal.com/docs/guide/function-invocation-methods)
  and [long-job example](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/long-training.py).
  It is separate from an empty GPU profiler capture, and no profiler retry was
  added. Preflight also uses spawn-only polling. The stdlib-importable
  `python -m cloud.modal_job --call-id fc-...` resumes the same call without
  importing the runner or allocating another job; Modal 1.6.3 requires hydration.
  Recovery was exercised on calls for `20261108T140200Z` and `20261108T142102Z`.
  Only literal builtin polling timeouts retry; other errors emit the call ID
  and recovery command, then propagate. The cause of the historical rejection
  remains unknown: the investigation is paused, not solved, and is not a
  current validation or release blocker. The archive identifies neither the
  rejected JWT nor a refresh failure; public call-history queries are
  unsupported, so attribution would need vendor validation/refresh records.
  Reopen vendor forensics only if a JWT/authentication failure recurs or the
  verified durable spawn/poll/reattach path is affected. Evidence and retained
  questions: `artifacts/modal-jwt-investigation-20261108T174201Z/` and
  VALIDATION.md.

- **Modal 1.6.3 SDK expired-cache fallback defect — open/reproduced.**
  `_AuthTokenManager._try_refresh_token` falls back to a cached token after a
  non-auth refresh failure even when that token is expired. Seven synthetic,
  real-SDK control-flow checks passed offline; this proves the defect exists,
  not that the historical job encountered it. Synchronous `AttemptAwait`
  normally refreshes header metadata between polls and also carries a separate
  attempt token. Evidence:
  `artifacts/modal-jwt-investigation-20261108T174201Z/reproduce.py`,
  `reproduction.json` and `reproduction.log`. This is local tracking only: no
  upstream issue has been filed or fixed, and there is no confirmed link to the
  historical JWT incident. No SDK patch or dependency upgrade was applied.

- **Historical CuTe DSL memcheck API-version failure (fixture fix verified).**
  See the fresh `20261108T200118Z` matrix below; the original failure remains evidence. In the GPU
  diagnostics matrix (run `20261108T192232Z`), `compute-sanitizer --tool memcheck`
  reported 18 CUDA API errors against CuTe DSL (`cuda_api_version_findings`:
  API 13041/13050 above driver 13030); its other three checks and all
  sanitizer checks on the other four backends pass, as do all five Nsight
  Systems and Nsight Compute captures. The Triton invalid-access control was
  correctly detected. The historical findings were not suppressed; the fresh
  fixture fix below does not clear broader CuTe DSL workloads. Summary:
  `artifacts/gpu-diagnostics-review-20261108T195500Z.json`.

- **Generated artifacts interrupted a Modal image upload.** The first full
  acceptance attempt after diagnosis failed before GPU execution because a
  concurrent local artifact log changed during image construction. The image
  now excludes `artifacts/` and explicitly includes only `artifacts/upstream.json`.
  Output logs and historical results are no longer build inputs; fresh reports
  are still written to the run volume. The next attempt passed image setup and
  preflight (run `20261108T140136Z`).

- **Current GPU acceptance passed; original collector cause is still unknown.**
  Full untraced run `20261108T140136Z` passed all 31 GPU tests and its automatic
  acceptance review. All five honest fixtures verified at provenance 1.0;
  all five Torch-op hacks verified at 0.0/reward -1; strict unmatched lists
  were empty. All required timings were positive. This verifies the new guard
  and evidence paths, not the cause of the historical zero baseline. The
  diagnosed empty-capture branch is unit-tested but did not reproduce on GPU.

- **CuTe DSL sanitizer API-version failure fixed for the captured fixture.**
  The first tool batch (`20261108T192232Z`) used CUDA bindings 13.5.0 and
  reported 18 memcheck errors: entry-point requests 13041/13050 exceeded the
  driver's 13030 API ceiling, despite five correct outputs. The GPU requirements
  now pin `cuda-bindings==13.3.0`; compiler/tileiras pins remain at 13.3.
  Fresh pilot `20261108T195956Z` reported zero errors and five output checks,
  while detecting the invalid-access control. Full matrix `20261108T200118Z`
  passed all 20 honest sanitizer checks. API errors were not suppressed; the
  original failure remains a regression fixture. This establishes only the
  shipped axpby fixture's captured coverage, not general CuTe memory safety.
- **Nsight importer and clock-control failures are resolved in the fresh matrix.**
  The importer needed `libdw.so.1`; the image's existing `libdw1` dependency
  supplies it. `ncu --clock-control none` avoids unavailable clock locking.
  Full run `20261108T200118Z` produced five real Nsight Systems reports/SQLite
  timelines and five selected-kernel counter captures, all reviewed as passed.
  Clocks remain unlocked; instrumented timings/counters never feed reward.
  Evidence and exact frozen source: `artifacts/h200-toolchain-fix-20261108T195717Z/`.
- **Step 2 diagnostic artifacts: GPU path not yet run on a GPU.**
  `--diagnostics` (`utils/profile_artifacts.py`) is covered on CPU, where it
  writes a truthful unavailable `report.json` and no trace, and by synthetic
  Torch-free tests. The GPU capture (Chrome trace, measured `kernels.json`,
  GPU/software metadata, CUPTI capability) is implemented and covered only by
  the skip-marked `tests.test_profile_artifacts.GPUArtifactTests`; it has not
  run on H200 yet. The diagnostic pass profiles only the generated model, with
  CPU activities, recorded shapes and memory events, so its kernel times are
  more perturbed than the reward profiler's and are diagnostics only. Artifacts
  live in the candidate-writable workdir and are never a trusted reward input.
