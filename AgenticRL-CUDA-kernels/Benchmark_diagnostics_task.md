# Task: trustworthy benchmarking and diagnostics for the kernel environment

You are implementing the next slice of `AgenticRL-CUDA-kernels/`, a CUDA/Triton kernel-generation RL environment. It already turns a candidate `model_new.py` into a reward. Before any RL trainer is built, the environment needs **stronger evidence that a candidate is really faster** and **diagnostics that explain why**.

Work **step by step**, in the order below. Finish each step completely (code, tests, docs, commit) before starting the next. Read `README.md`, `VALIDATION.md` and `KNOWN_ISSUES.md` first; they describe the current behaviour and its limits.

---

## What exists today (verified against the code)

- **The profiler sums kernel durations; it does not measure elapsed time.** `utils/profiling.py:49-63` warms up, records CUDA events with `torch.profiler`, and reports `sum(event durations) / iters`. With overlapping or multi-stream GPU work, that sum is not wall time, and profiler instrumentation itself adds overhead.
- **Measurement order is fixed.** The generated model runs first, then eager, then `torch.compile` (`utils/profiling.py:114-117`). There is a single round with no variance statistics. Compile/autotune happens inside the warmup.
- **No diagnostics are exported.** There are no trace exports, repeated-sample statistics, Nsight workflows or hardware-counter reports.
- **`run_task` keeps only four measurements.** It parses the **last stdout line** of the profiling stage as JSON and keeps only `t_eager_us`, `t_compile_us`, `t_gen_us` and `provenance` (`utils/rollout_env.py:148-155`). Per-kernel times and the `provenance/*` category shares are discarded.
- **Provenance is name-based.** `utils/provenance.py:12-22`: authored names count as agent; `aten`/`at::native` as torch; `cublas|cutlass|cudnn` as library on CUDA but torch on Triton; memcpy/memset as memory (excluded); anything unknown counts as torch, so the check fails closed.
- **Nothing has run on a GPU yet.** Real GPU timings and real profiler kernel names have never been observed (`KNOWN_ISSUES.md:3-8`). Every GPU cell in the README says *not measured*.
- **The stock task is tiny.** It is a single 1 × 128 tensor pair (`model.py:19-22`), which cannot support general optimisation claims.

## Ground rules (apply to every step)

1. **Never fabricate numbers.** Without a measurement, the value is `null` / *not measured* with a reason. Every number written into docs carries the exact command that produced it. Synthetic fixtures must be labelled synthetic.
2. **Keep diagnostics out of the reward.** Profiler instrumentation, trace capture and Nsight replay perturb execution, so their timings must never silently become training rewards (PyTorch documents profiler overhead: https://docs.pytorch.org/tutorials/beginner/profiler.html). New modes are **opt-in and off by default**.
3. **Don't change existing semantics.**
   - The upstream stdout line `Torch Baseline: …us, Torch Compile: …us, CUDA Extension: …us` stays byte-identical (`tests/test_backend_provenance.py:76`).
   - `t_gen_us`, `t_eager_us`, `t_compile_us`, `provenance` and the reward rules keep their current meaning.
   - The profiling `--json` output stays the last stdout line.
   - Add new fields; never rename, remove or redefine existing ones.
   - Do **not** switch the reward to the new timings in this task. Propose that as a decision for the user.
4. **Don't touch protected upstream files.** That means `model.py`, `model_new.py`, `binding.cpp`, `binding_registry.h`, `kernels/axpby.cu`, `kernels/axpby_binding.cpp` (checksummed in `artifacts/upstream.json`; enforced by `tests/test_backend_provenance.py:18`) and `SKILL.md` (upstream text).
5. **Keep the pure-Python modules Torch-free.** `utils/config.py`, `reward.py`, `grpo.py` and `rollout_env.py` must import under `python -S` without Torch (`tests/test_config_reward_grpo.py:62`). Put new statistics, report-assembly and table-aggregation logic in Torch-free helpers so it can be unit-tested without a GPU. Import Torch lazily inside GPU code paths, as `utils/profiling.py` already does.
6. **Results stay flat and JSON-safe.** `run_task` returns scalar or string values only, with stable key names. Detailed data goes into artifact files, and the result carries their paths. `tests/test_rollout_env.py:36-49` pins the profiling argv (`:44`) and `len(result) == 18` (`:48`). Update that test deliberately when adding keys; prefer asserting the explicit key set over a count. With all new modes off, the default argv must stay unchanged.
7. **No new Python dependencies** beyond `torch>=2.3` and `triton>=3.0` (NumPy only for the CPU Triton interpreter). Nsight Systems, Nsight Compute and Compute Sanitizer are external binaries: detect them at runtime and report them as unavailable when missing.
8. **Every `RolloutConfig` field needs a `--kebab-case` CLI flag** (`tests/test_config_reward_grpo.py:35`). Presets are applied before overrides.
9. **Attribution stays conservative.**
   - Unknown kernel names keep counting as torch.
   - Don't add a blanket `triton_` rule: Inductor emits that prefix too.
   - Name matching is a heuristic, not proof of authorship. Say so wherever results are reported.

## Environment facts

- **Develop on Linux or WSL2.** `run_task` uses POSIX process groups (`os.killpg`), so five process tests fail on native Windows.
- **The current dev host has no GPU, `nvcc`, Ninja, `nsys`, `ncu` or `compute-sanitizer`.** CUDA builds and every GPU measurement need a GPU host.
- **Test commands** (run inside `AgenticRL-CUDA-kernels/`):
  ```bash
  python3 -m unittest discover -v                  # system Python, no Torch: fast; currently 43 tests, 11 skipped
  .venv/bin/python -m unittest discover -v         # CPU Torch + Triton interpreter: slow (~10 min)
  .venv/bin/python -m scripts.validate --device cpu --timeout-s 600
  ```
  On a GPU host:
  ```bash
  python -m unittest -v tests.test_provenance.GPUIntegrationTests
  python -m scripts.validate --device cuda
  ```

## How to work through each step

1. Re-read the code the step touches. State a short plan: files, new keys and CLI flags, tests.
2. Implement the step. Keep diffs focused and follow the existing style (small modules, docstrings that state limits).
3. Add tests:
   - Torch-free unit tests for pure logic.
   - CPU tests showing the *unavailable* path.
   - GPU tests that skip explicitly without CUDA, with the reason in the skip message.
4. Run both local test suites; both must be green. Existing tests may change only where this brief says so.
5. Update `README.md`, `VALIDATION.md` and `KNOWN_ISSUES.md` wherever behaviour, commands or limits changed.
6. Commit the step as one focused commit (don't push).
7. Report back:
   - what changed
   - the raw test summary lines (`Ran N tests…`, `OK (skipped=…)`)
   - what could not be verified, and why
   - any open decisions

Then continue to the next step.

**Stop and ask the user** instead of guessing when:
- a step needs a GPU and none is available;
- a change would alter reward semantics or existing result keys;
- a protected file would need editing;
- verification tolerances would need to change;
- you hit a step marked **ASK**;
- you've spent more than ~30 minutes on one blocker (log it in `KNOWN_ISSUES.md` first).

---

## Step 1: Repeatable benchmark mode *(priority: first)*

**Goal:** answer "is the generated implementation actually faster, outside profiler instrumentation?"

Add a benchmark mode, for example `utils/benchmark.py`, run as `python -m utils.benchmark --json` (or an opt-in flag on `utils.profiling`), plus Torch-free statistics helpers. It must provide:

- **Configurable phases.** Use `--warmup-iters`, `--rounds` and `--iters-per-round`. Pick conservative defaults that fit the 180 s task deadline, make rounds a multiple of 3, and document them.
- **Compile and autotune kept out of the measurements.** Run an explicit *prepare* phase per implementation first (first `torch.compile` call, Triton JIT plus autotune, CUDA extension load) and record its duration separately, e.g. `bench/<impl>/prepare_s`. Then warm up, then measure. Prepare time never enters steady-state statistics.
- **Two clearly labelled timings per round,** for each of `eager`, `compile` and `generated`:
  - **CUDA-event elapsed time.** Record `torch.cuda.Event(enable_timing=True)` before and after the round's iterations on the current stream, then synchronize. Store it as e.g. `bench/<impl>/event_us_*`.
  - **Synchronized host wall-clock latency.** Run `torch.cuda.synchronize()`, start `time.perf_counter()`, run the iterations, synchronize again, stop. Store it as e.g. `bench/<impl>/wall_us_*`. This includes CPU dispatch cost; say so in the field docs.
- **Statistics over rounds, not one average:** n, median, min, max, p25/p75 (IQR) and coefficient of variation. Keep the raw per-round samples for the report (Step 2).
- **Balanced execution order.** Rotate the three implementations each round (ABC, BCA, CAB, …) so each occupies each position equally often. Record the order and the seed.
- **A stream/device contract.** Document it in `README.md` and `SKILL_TRITON.md`. `ModelNew.forward` must enqueue all device work on `torch.cuda.current_stream()` of the inputs' device, or make that stream wait on any side streams before returning. Report the event-vs-wall ratio for the generated implementation as a diagnostic, so unjoined side-stream work is visible rather than silently mismeasured.
- **The device actually used:** index, name, compute capability, and UUID when available.
- **CPU and unavailable behaviour.** Emit `bench/available: false` and a `bench/unavailable_reason`, with every timing `null`. Never produce timings from the CPU interpreter.

### CUDA graphs: optional replay timing

**Is replay needed?** Not for the reward, and not by default. It is a diagnostic: it separates *GPU execution time* from *CPU dispatch and kernel-launch overhead*. That overhead dominates launch-bound tasks such as the stock 1 × 128 case, where the normal timings mostly measure how many kernels each path launches. For large bandwidth- or compute-bound shapes, graph and non-graph timings should converge. Report both and let the data decide.

Add it behind an opt-in flag, e.g. `RolloutConfig.cuda_graphs: bool = False` (CLI `--cuda-graphs`):

- **Capture after prepare.** For each implementation, after its prepare phase:
  1. Warm up on a side stream, then make the current stream wait on it.
  2. Copy inputs into **static buffers**.
  3. Capture one forward (or K forwards, to amortise replay launch) into a `torch.cuda.CUDAGraph`.
  4. Time `graph.replay()` rounds with CUDA events into `bench/<impl>/graph_us_*`, using the same statistics and rotation as above.
- **Capture failure is a result, not a crash.** Record `bench/<impl>/graph_capturable: false` plus `bench/<impl>/graph_error`. Typical causes:
  - kernels launched on the legacy default stream instead of the current stream
  - host synchronisation inside `forward` (`.item()`, `.cpu()`, `torch.cuda.synchronize()`)
  - CPU control flow that depends on tensor values
  - Triton autotune or `torch.compile` compilation still running (they must finish in prepare)

  A failed capture never changes the reward.
- **Check correctness on replay.** Copy fresh inputs into the static buffers, replay, and compare the static output with the eager reference (same tolerances as verification). A mismatch means CPU-side state was baked into the graph. Report it as `bench/<impl>/graph_replay_ok: false`.
- **Compare like for like.**
  - Only report a graph comparison when all three implementations captured successfully. Never compare a graph-replayed candidate against non-graph baselines.
  - `torch.compile(mode="reduce-overhead")` uses CUDA graphs internally, but the existing baseline uses the default mode. If you add a reduce-overhead baseline, give it its own keys (e.g. `bench/compile_reduce_overhead/*`). Leave `t_compile_us` alone.
- **Keep profiling non-graph.** Provenance and the Step 2 per-kernel tables stay non-graph, so attribution keeps working on ordinary kernel launches.
- **ASK** before making graph-capturability part of the policy contract (`SKILL_TRITON.md`) or using graph-replay timings in the reward.

**Integration.** Add an opt-in `RolloutConfig` field (e.g. `benchmark: bool = False` plus the round settings) that makes `run_task` run the benchmark as a **separate stage** after profiling, under the same deadline. Return flat `bench/*` keys. With the flag off, the stage argv, result keys and reward are unchanged.

**Tests:**
- Statistics on fixed samples.
- Rotation is balanced for rounds = 3k.
- Prepare time is excluded.
- The CPU path returns *unavailable* without importing Torch where possible.
- `run_task` with the flag off still produces exactly the old key set; with it on, it produces old keys plus `bench/*`, all flat (`assert_flat`).
- GPU-only (skipped here): samples are finite and positive, `n == rounds`, and the order is balanced.
- CUDA graphs:
  - With `--cuda-graphs` off, no `graph_*` keys appear and nothing is captured.
  - On CPU, graph timing is reported unavailable.
  - GPU-only: honest fixtures capture and replay correctly, and a fixture that calls `.item()` inside `forward` reports `graph_capturable: false` instead of crashing.

**Done when** both local suites are green, the default behaviour is byte-for-byte unchanged, and the docs explain both timings and the stream contract.

## Step 2: Actionable profiling artifacts *(priority: first; implemented, CPU-validated and H200-validated for the Triton fixture in run 20261108T225150Z — see VALIDATION.md)*

**Goal:** answer "which operators and kernels run, and where are the copies, allocations and launch gaps?", and keep the details instead of discarding them.

Add an opt-in diagnostic mode (e.g. `RolloutConfig.diagnostics: bool = False`; CLI `--diagnostics`). It runs its **own** profiler pass, separate from the Step 1 benchmark and from the reward measurement. It writes into the task workdir, e.g. `<workdir>/.artifacts/<run_id>/`:

- **`trace.json`:** a Chrome-format CPU + CUDA timeline (`torch.profiler` `export_chrome_trace`).
- **`kernels.json`:** a per-kernel table with name, calls, total µs, mean µs and the existing provenance class (`utils/provenance.py`, unchanged). Keep the aggregation in a Torch-free helper.
- **`report.json`**, containing:
  - `schema_version`, `run_id` and timestamps
  - task id and backend
  - input shapes and dtypes from `get_inputs()`
  - GPU metadata (name, compute capability, memory, UUID) and software versions (driver, CUDA runtime, torch, triton, Python)
  - SHA-256 of `model.py`, `model_new.py`, every file under `kernels/` and the trusted `utils/` harness
  - the benchmark configuration, raw Step 1 samples and statistics (when enabled), including CUDA-graph capture status, errors and replay samples when `--cuda-graphs` is on
  - provenance totals and shares
  - a **capability report:** which of `nvidia-smi`, `nvcc`, `ninja`, `nsys`, `ncu`, `compute-sanitizer` are on `PATH`, and whether CUPTI produced CUDA events. Each entry is available, or unavailable with a reason.

**Changes to `run_task`:**
- Always return the provenance shares as flat keys: `env/provenance_agent`, `env/provenance_library`, `env/provenance_torch`, `env/provenance_memory` (`null` when unmeasured).
- With diagnostics on, also return `artifact/report` and `artifact/trace` as path strings, or `null` with a reason.
- Document that `artifact/*`, like `log/*`, is dropped before `wandb.log`.
- Never read artifacts back to compute a reward. The agent can modify files in its workdir.

**Tests:**
- Kernel-table aggregation and provenance classes on synthetic event lists (labelled synthetic).
- `report.json` round-trips through `json.dumps(allow_nan=False)`.
- Source hashes change when a kernel file changes.
- On CPU, diagnostics produce a report saying the GPU, trace and timings are unavailable, with no fabricated trace.
- Returned paths exist, and the result stays flat.

**Done when** a CPU run with `--diagnostics` writes a truthful *unavailable* report, and the GPU code paths are implemented and covered by skip-marked tests.

## Step 3: Real-GPU acceptance evidence *(needs a GPU host)*

**Goal:** prove the slice on hardware and pin the runtime that produced the proof.

1. **Pin the runtime used for the run.** Record the exact Python, torch, triton, CUDA toolkit, driver and GPU, as a lockfile or image definition checked into this folder.
2. **Extend `scripts.validate` and `tests.test_provenance.GPUIntegrationTests`** to run the honest CUDA, honest Triton and Torch-op hack fixtures with benchmark and diagnostics on. Write results to `artifacts/` with the commands alongside.
3. **Record the real `ctx.events()` kernel names** for each fixture, and check them against the classifier. This closes the open item in `KNOWN_ISSUES.md:3-8`. If names don't match expectations, report them; don't loosen unknown-event handling.
4. **Run once with `--cuda-graphs`.** Record capture status, replay correctness, and graph-vs-non-graph timings for all three implementations of each fixture. Capturability is reported, not required, unless the user decides otherwise.

**Acceptance criteria (all must hold):**
- Honest CUDA and Triton implementations pass correctness **and** provenance.
- The ordinary Torch-op substitution still passes correctness and **fails** provenance.
- Benchmark samples are finite, positive and reproducibly recorded (raw samples plus config in `report.json`).
- Traces contain real GPU kernel events, not only host events.
- CPU-only runs still report GPU measurements as unavailable.
- The existing stdout format and the reward-boundary tests are unchanged.
- Missing tools or permissions produce explicit *unavailable* results, never made-up metrics.

Fill the README's *not measured* cells only from this run's output, with the command next to each number. Update `VALIDATION.md` and `KNOWN_ISSUES.md`.

**If no GPU is available:**
- Stop after making everything runnable.
- Give the user one copy-paste command block for a GPU host.
- Don't write Modal or orchestration code in this step.

**Do not start Steps 4–7 until Step 3's evidence exists.**

---

## Step 4: Nsight Systems, Nsight Compute, Compute Sanitizer *(implemented and H200-validated for the stock fixtures — see VALIDATION.md)*

Add a diagnostics entry point, e.g. `python -m utils.diagnose {nsys,ncu,sanitizer} [--kernel REGEX] [--out DIR]`. It runs the trusted harness under the tool inside the workdir, with its own timeout, and saves the tool's outputs as artifacts referenced from `report.json`. None of these results ever feed the reward.

| Tool | Question it answers | Notes |
|---|---|---|
| **Nsight Systems** (`nsys`) | Is performance limited by CPU dispatch, synchronization, transfers or GPU idle time? | Whole-run timeline; summarise idle gaps and transfer time. |
| **Nsight Compute** (`ncu`) | Is a *selected* kernel limited by memory bandwidth, occupancy, registers or compute throughput? | Target one kernel (`--kernel` plus a small launch count). Use it after Step 3 or `nsys` identifies a bottleneck. |
| **Compute Sanitizer** (`compute-sanitizer`) | Are there invalid accesses, races or synchronization errors? | Not a performance tool. Run memcheck, racecheck, synccheck and initcheck on the verification path. Honest fixtures must be clean before Step 5. |

Report a missing binary or a counter-permission error (e.g. `ERR_NVGPUCTRPERM`) as unavailable with the tool's message. Build parsers only from **real tool output captured on the GPU host**; never from invented samples.

**Tool replay and CUDA graphs.** These tools have their own replay mechanics, separate from Step 1's graph replay:

- **Nsight Compute replays each profiled kernel several times** to collect counters, saving and restoring device memory between passes (`--replay-mode`). Its durations are therefore not latencies; use it for *why*, never for *how fast*.
- **Graphs need explicit options.** When profiling graph-replay runs, choose whether to profile individual graph nodes or the whole graph (Nsight Compute `--graph-profiling`, Nsight Systems `--cuda-graph-trace`). By default, a graph launch can appear as one opaque unit.
- **Flag names vary by version.** Confirm them against `--help` for the installed versions and record those versions in `report.json`.

## Step 5: Representative workloads

Add separate task fixtures, e.g. under `tasks/<name>/`, each shaped like a workdir with its own `model.py`. The root `model.py` stays untouched. Cover:

- sizes from launch-bound (the 1 × 128 case) up to bandwidth-bound (millions of elements)
- `float32`, `float16` and `bfloat16`
- at least one reduction (e.g. row softmax or layernorm)
- a GEMM with an epilogue (Level-2 style; the deferred `gemm_div_sum_scale` via `tl.dot` is a good candidate)

Each fixture needs an honest Triton solution (CUDA where practical) and a Torch-op hack variant, so the provenance gate is exercised per task. Extend `scripts.validate` to run the suite.

**ASK** before changing verification tolerances for reduced precision. The upstream checks use atol/rtol = 1e-2.

## Step 6: GPU contention control

Serialize the **measurement** stages (profiling, benchmark, diagnostics) per physical GPU, keyed by GPU UUID, for example with a file lock in a configurable directory. Compile and verification may still run concurrently. Record the actual device (`meta/gpu_uuid`, `meta/gpu_name`) and the lock wait time. Make sure locks are released on crash and timeout.

Separate workdirs prevent build collisions, not timing interference.

**ASK** how lock wait interacts with the task deadline. A candidate must not get `reason: "timeout"` because another rollout held the GPU.

## Step 7: Execution isolation (design first)

Before autonomous candidates run unattended, the environment needs:
- a restricted container (non-root, no network, writable workdir only)
- read-only trusted `utils/` and harness
- CPU, memory, PID and disk limits
- GPU scoping
- an outer container timeout longer than the inner deadline

POSIX process groups alone are not a security sandbox. Write a short design note first (local Docker vs Modal sandbox, the mount layout and the limits), then **ASK** the user to choose before building a runner.

## Out of scope for this brief

The agent loop, context-budget enforcement, Modal dispatch, checkpointing and experiment tracking (MLflow/W&B) all come after the evaluator is trustworthy. Do not add tracking-library imports.

**Overall sequence:** reliable benchmarking → trace export → real-GPU validation → Nsight diagnostics and sanitizer checks → broader workloads → contention control and isolated rollouts → training.
