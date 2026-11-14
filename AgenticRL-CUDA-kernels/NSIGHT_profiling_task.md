# Task: Nsight Systems and Nsight Compute commands for the task suite (profilers, opt-in)

You are working in `AgenticRL-CUDA-kernels/` (WSL2; GPU work on a Modal H200). Read `CLAUDE.md`,
`AGENTS.md`, the Ground rules at the top of `Benchmark_diagnostics_task.md`,
`Architecture-decision/0005-backend-diagnostic-controls.md`, the README sections "GPU tool diagnostics
(opt-in)", "Diagnostic profiling artifacts (opt-in)", "CUDA-graph replay" and "GPU trace run", and
`VALIDATION.md` §"H200 sanitizer and Nsight fixes — 20261108T200118Z" before writing anything.
**Do not commit and do not launch any GPU run**: Claude reviews, commits from the Windows checkout and
launches approved H200 runs. Finish cleanly, with both suites green and nothing in flight.

## What exists (build on it; do not rebuild it)

The repo already has three opt-in diagnostic runners, validated on the H200 on 2026-11-08 for the stock
1 × 128 AXPBY fixture across all five backends (20/20 sanitizer checks clean, 5/5 Nsight Systems
timelines, 5/5 Nsight Compute captures; run `20261108T200118Z`):

- `utils/diagnose.py`: `run_diagnostic(tool, workdir, out, ...)` runs the trusted harness worker
  (`utils/diagnostic_worker.py`) under one tool. The worker wraps exactly one generated forward in the
  NVTX range `agenticrl_generated` (`RANGE`). Today's commands (`tool_command`):
  - **nsys:** `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --discard-environment=true
    --capture-range=nvtx --nvtx-capture=agenticrl_generated --capture-range-end=stop --export=sqlite
    --output <out>/capture <worker>`
  - **ncu:** `ncu --clock-control none --nvtx --nvtx-include agenticrl_generated/ --kernel-name-base
    demangled --kernel-name regex:<authored names> --launch-count 1 --set basic --export <out>/capture
    <worker>` (`--clock-control none` was required on the H200; keep it)
  - **sanitizer:** `compute-sanitizer --tool <memcheck|racecheck|synccheck|initcheck>
    --error-exitcode 86 --target-processes application-only --log-file ... <worker>`
- `utils/diagnostic_review.py` reads the tool output into `report.json` (fail-closed: unrecognised
  output is `unavailable`, never a pass); `scripts/review_diagnostics.py` and `scripts/recover_nsys.py`
  (re-imports `.qdstrm` files with `QdstrmImporter`) are the CLI; `scripts/diagnose_gpu.py` and
  `cloud/modal_runtime.py --diagnose` run the matrix on Modal; `tests/test_diagnose.py`,
  `tests/test_diagnostic_review.py`, `tests/test_modal_diagnostics.py` cover them with fake tool output.
- Newer: `scripts/graph_trace.py` + `utils/graph_trace_worker.py` + `scripts/render_nsys_timeline.py`
  profile CUDA-graph capture and replay under `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`
  with one NVTX range per phase, and render SVG timelines from the SQLite export (run
  `graph-trace-20261112T220601Z`, 24 traces). `cloud/modal_graph_trace.py` runs it on an H200.
- The Modal image (`cloud/modal_runtime.py`) has `nsys` 2026.2.1.0 at `/usr/local/bin/nsys`, `ncu` at
  `/usr/local/cuda/bin/ncu`, `compute-sanitizer`, `libdw1` (for the nsys importer), and
  `cuda-bindings==13.3.0`. The dev host has none of these tools and no GPU.
- Environment variables reach a Modal diagnostic job only through `DIAGNOSTIC_ENV` in
  `cloud/modal_runtime.py` (`forwarded_environment`); `CUTE_DSL_LINEINFO` is not in it yet (ADR 0005).
- The Stage 6 technical report and README already show Nsight timelines of graph replay; the per-kernel
  Nsight Compute view is the gap this task fills.

**Ground rules that bind every command here:** diagnostics are opt-in, off by default, and never feed a
reward; the profiling `--json` line and the upstream timing line are unchanged; results stay flat and
JSON-safe; no new Python dependency; `nsys`, `ncu` and `compute-sanitizer` are detected at runtime and
reported `unavailable` when missing; never fabricate a number; keep the pure-Python modules importable
under `python -S` without Torch.

## Goal

Make Nsight Systems and Nsight Compute usable on **any task in the catalog** (not only AXPBY), with
commands an engineer can copy, and with NVTX annotation that lets a timeline be split by harness stage
and by kernel phase. Deliver three things: (1) a set of **named profiling profiles** (command presets),
(2) **NVTX ranges** in the harness and in the honest fixtures where they clarify a kernel, and (3) a
**per-kernel metrics summary** from Nsight Compute next to the suite rows. Everything is diagnostic.

## Part 1: Nsight Systems profiles

Add to `utils/diagnose.py` a table of named **profiles** that `tool_command("nsys", ...)` selects with a
new `--profile` option (default `trace`, which must reproduce today's command byte for byte so existing
tests and evidence stay valid). Each profile is one `nsys profile` argument list; document the cost of
each in its docstring. Proposed set (verify every flag against `nsys profile --help` of **2026.2.1**, the
pinned image version, before using it; drop or rename anything that version does not accept):

| Profile | Adds over `trace` | Use |
|---|---|---|
| `trace` (default) | nothing: `--trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=agenticrl_generated --capture-range-end=stop --export=sqlite` | the existing validated capture of one generated forward |
| `stages` | `--nvtx-capture` widened to the whole rollout (`agenticrl_rollout`, see Part 2) and `--trace=cuda,nvtx,cublas,cudnn,osrt` | which library kernels the eager and `torch.compile` baselines launch; where compile, verify, profile time goes |
| `pytorch` | `--pytorch=functions-trace,autograd-shapes-nvtx` (nsys's own PyTorch NVTX injection) | module-level attribution of the eager baseline without editing model code |
| `backtrace` | `--cudabacktrace=all --python-backtrace=cuda` | which Python line launched each kernel (high overhead; never on a timed row) |
| `gpu-metrics` | `--gpu-metrics-devices=<assigned device index>` (`--gpu-metrics-device` in older versions; check the spelling) | SM activity, Tensor Core activity, DRAM bandwidth over time alongside the kernels |
| `graphs` | `--cuda-graph-trace=node` | per-node kernels inside `cudaGraphLaunch` (the existing graph-trace command) |

Rules: profiles compose (`--profile stages,gpu-metrics`); `--sample=none --cpuctxsw=none` stay on
unless a profile needs CPU sampling; the output path stays `<out>/capture` with `--export=sqlite`; the
device index for `--gpu-metrics-devices` comes from the lock's device discovery (`utils/gpu_lock.py`
maps the logical ordinal through `CUDA_VISIBLE_DEVICES`), never hard-coded 0. Record the exact argv in
`report.json` as today. Extend `review_nsys` to summarise, per NVTX range: kernel count, GPU busy time,
library vs authored kernel time (reuse `utils/provenance.py` classification by name), and, when
`gpu-metrics` ran, the mean SM-active and DRAM-throughput percentages inside the generated range (read
from the SQLite `GPU_METRICS` table; say exactly which table and columns, and mark the fields null when
the table is absent).

## Part 2: NVTX ranges in the harness and the fixtures

- **Harness stages.** In `utils/diagnostic_worker.py`, wrap the whole diagnostic run in
  `agenticrl_rollout` and each phase in its own range: `agenticrl_warmup` (the warm-up calls, so a
  `--nvtx-capture=agenticrl_generated` run still excludes them), `agenticrl_eager`, `agenticrl_compile`,
  `agenticrl_generated` (unchanged name, unchanged content: exactly one forward). Keep `RANGE` and the
  existing capture semantics; add constants for the new names. `torch.cuda.nvtx` is the only API used;
  on CPU the worker must not import it.
- **Kernel phases in honest fixtures (optional, per task).** Where a task's forward has distinct phases
  (for example `flash_attention_fwd`: QKᵀ, online softmax, PV; `moe_dense_gemm`: FC1, SwiGLU+alpha,
  FC2; `lora_fused_sgmv`: shrink, expand), add `torch.cuda.nvtx.range(...)` blocks **in the Python
  wrapper only**, named `<task>/<phase>`. Rules: never inside a `@triton.jit`, `@ct.kernel` or CUDA
  kernel; never on the reward-profiled path unless the range is free of synchronisation (an NVTX push/pop
  is host-side and cheap, but say so and measure it once: compare `t_gen_us` with and without the
  ranges on one task and report the difference); if a task's wrapper is a single launch, add nothing.
  Start with three tasks and list the rest as follow-ups.
- **Fixture provenance check.** NVTX ranges must not change what the provenance classifier sees: the
  kernel names are unchanged, so the authored share is unchanged. Add a test that classifies a synthetic
  kernel list with and without NVTX events and gets the same shares.

## Part 3: Nsight Compute per-kernel metrics for honest kernels

Extend the `ncu` path so it runs on a **task workdir** (today it is exercised on AXPBY) and selects the
task's authored kernels by name (`kernel_selector` already builds the regex from
`agent_kernel_names`). Add `--set` presets: `basic` (today), `roofline` (`--set full` is too slow;
use `--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy`, verify the section
names in `ncu --list-sections` for the image's ncu version), and `launch` (`--section LaunchStats`).
Keep `--launch-count 1 --clock-control none`. Extend `review_ncu` to pull, per kernel: duration, SM
throughput %, DRAM throughput %, achieved occupancy, registers per thread, shared memory per block,
grid and block sizes, and write them as `ncu/<kernel>/<metric>` fields plus a Markdown table in
`report.json`'s companion `ncu-summary.md`. Numbers come from the `--export` report via
`ncu --import <file> --csv --page raw` (or `--page details`); say which, and parse fail-closed.

Then add a suite hook: `scripts/validate.py --suite ... --ncu <preset>` runs `run_diagnostic("ncu")`
on **honest rows only**, after the row's normal evaluation, in the same workdir, under the GPU lock,
and attaches `ncu/*` fields to the row as diagnostics (never to `reward/*`; add the row keys to
`tests/support.py::DEFAULT_RESULT_KEYS` deliberately and ADR 0002 `result:` rows). Skip, with a reason
field, when `ncu` is missing, when the row is not honest, or on CPU.

## Part 4: Compute Sanitizer on the task suite (same hook, smaller)

`--sanitizer <check>` on `scripts/validate.py` runs `run_diagnostic("sanitizer", ...)` on honest rows
the same way and records `sanitizer/<check>/status` and the error count. The Stage 5 and Stage 6
kernels and the nine cuTile candidates have never been under the sanitizer; prepare (do not launch) one
request covering them.

## Part 5: commands, docs and evidence

- Add a README subsection "Nsight profiles" under "GPU tool diagnostics (opt-in)" with one copy-paste
  block per profile, local and Modal forms, for example:
  ```bash
  # one generated forward of a task under Nsight Systems (existing default profile)
  python -m utils.diagnose nsys --backend triton --workdir <task workdir> --out <dir>
  # whole rollout with library tracing and GPU metrics
  python -m utils.diagnose nsys --profile stages,gpu-metrics --backend cuda --workdir <task> --out <dir>
  # Python backtraces for every kernel launch (slow; never on a timed row)
  python -m utils.diagnose nsys --profile backtrace --backend triton --workdir <task> --out <dir>
  # Nsight Compute roofline sections for the task's authored kernels
  python -m utils.diagnose ncu --preset roofline --backend cuda --workdir <task> --out <dir>
  # the suite, honest rows only, with ncu roofline and memcheck attached as diagnostics
  python -m scripts.validate --suite stage6 --device cuda --ncu roofline --sanitizer memcheck
  # on Modal
  modal run --detach cloud/modal_runtime.py --diagnose --diagnostic-tools nsys,ncu ...
  ```
  State the overhead of each profile from one measurement, and that none of it enters a reward.
- `VALIDATION.md`: a new section with the commands and the fake-output tests; `KNOWN_ISSUES.md`: tool
  limitations (clock control, `--pytorch` needs a Torch build with NVTX hooks, `gpu-metrics` needs
  permission on some hosts, the historical empty-capture issue is still unreproduced).
- Evidence bundle `artifacts/runs/nsight-local-<UTC>/` with both suite logs and the prepared Modal
  requests (`launched: false`), as `artifacts/runs/cutile-local-20261113T014034Z/` does.

## Tests (no GPU)

- Every profile's argv is pinned by a test (the default byte-identical to today's).
- `review_nsys` and `review_ncu` are tested on **synthetic** SQLite files and CSV exports with the
  tables and columns you rely on (label them synthetic; `tests/test_graph_trace.py` shows the pattern).
- The validate hooks are tested with `run_diagnostic` mocked: honest rows get `ncu/*` and
  `sanitizer/*`, other rows and CPU runs get the skip reason, `reward/*` is identical with and without
  the hooks (`tests/test_cuda_graph_guardrails.py` shows how to pin "diagnostics never change a reward").
- NVTX: the diagnostic worker's range names and nesting, asserted with `torch.cuda.nvtx` patched.
- Both full suites green; report both summaries.

## Stop and ask before

Changing any tool flag that the 2026-11-08 evidence depends on (the default `trace` profile and the
`ncu basic` command), adding a range inside a timed reward path without the one-task overhead
measurement, editing protected upstream files, or adding a dependency. Launch nothing.

## Report back

Files changed; the profile table with the verified flags for nsys 2026.2.1 and the image's ncu version
(and which flags you could not verify locally); the NVTX ranges added and the measured overhead on one
task; the `ncu/*` and `sanitizer/*` field list; the prepared Modal commands with row counts and
estimates; both suite summaries; open questions. Nothing committed, nothing launched.
