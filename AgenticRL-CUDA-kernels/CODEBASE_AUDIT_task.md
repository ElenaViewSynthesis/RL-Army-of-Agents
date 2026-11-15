# Comprehensive Codebase Audit — Agentic RL CUDA Kernel Research Framework

## Role and Objective

Act as a **Principal GPU Kernel Engineer, CUDA/Triton Performance Specialist, Distributed Reinforcement Learning Research Engineer, and Research Software Auditor**.

Conduct a rigorous, repository-wide technical audit of my GPU kernel optimisation and agentic reinforcement learning research framework.

**Repository:** https://github.com/ElenaViewSynthesis/RL-Army-of-Agents

**Target subdirectory:** `AgenticRL-CUDA-kernels/`

**Branch:** `main`, unless a different active development branch contains more recent relevant implementations.

The objective is to identify **every substantive implementation gap, incomplete feature, incorrect abstraction, nonfunctional execution path, missing GPU kernel, weak test, unsubstantiated performance claim, and research reproducibility deficiency**.

Pay particular attention to all `model.py` and `model_new*.py` implementations, CUDA/Triton kernels, RL optimisation routines, benchmark validation, GPU execution, and the integration between these components.

This is an evidence-driven engineering audit, not a superficial README review.

Do not modify the repository during the initial audit.

**State of the repository when this brief was last updated (2026-11-15, commit `926d49c`).** The parts of this brief
written for an early, AXPBY-only repository have been brought up to date. The root workdir still holds the protected
AXPBY example, but `tasks/` now holds 39 kernel tasks in Stages 0–7 (catalog: `utils/task_catalog.py`), including
custom GEMM kernels, with H200 evidence in `artifacts/runs/`. There is still no trainer: the repository is an RL
*environment* that scores kernel candidates, not a fine-tuning loop. Every statement below about the current state is a
claim for the audit to verify, not a conclusion.

---

## 1. Repository-Wide Inspection

Recursively inspect the entire `AgenticRL-CUDA-kernels/` directory.

Start by establishing the exact Git commit SHA and generating a complete repository file inventory.

Review the following components and their dependencies.

### A. Model implementations

Inspect at minimum:

- `model.py`
- `model_new.py`
- `model_new_triton.py`
- `model_new_tilelang.py`
- `model_new_cutedsl.py`
- `model_new_cutile.py`

Then inspect every task under `tasks/<name>/`: its trusted `model.py` reference and its fixtures. Each task ships
honest solutions (`model_new_cuda.py`, `model_new_triton.py`, and where present `model_new_cutedsl.py`,
`model_new_cutile.py`, `model_new_tilelang.py`), a reward-hack control (`model_new_hack.py`, which must pass
verification and fail provenance) and a wrong-output control (`model_new_wrong.py`). Some stages add more controls:
`model_new_wrong_ancestor.py` (Stage 5), `model_new_wrong_padding.py`, `model_new_wrong_skip.py` and the
diagnostic-only `model_new_fp16_intermediate.py` (Stage 6), and the alternate `chain_replay` solution of
`eagle_gdn_tree_recurrence`. `fp8_gemm` and `int8_gemm` are contract-only, with no fixtures.

Identify all additional `model.py`, `model_new.py`, or other candidate implementations in nested directories.

Distinguish canonical entrypoints from generated candidate snapshots and archived experiment artifacts. In particular,
`artifacts/sources/` holds frozen copies of the source that GPU runs used, and `artifacts/runs/*/source/` and
`artifacts/runs/*/suite/**/model_new.py` are run snapshots; none of these is canonical.

### B. GPU kernel implementations

Inspect:

- `kernels/axpby.cu`
- `kernels/axpby_binding.cpp`
- `binding.cpp`
- `kernels/triton/`
- `kernels/tilelang/`
- `kernels/cutedsl/`
- `kernels/cutile/`
- Each task's kernels: `tasks/<name>/kernels/<name>.cu` with its `*_binding.cpp`, and
  `tasks/<name>/kernels/{cuda,triton,cutedsl,cutile}/`
- Backend wrappers, dispatchers, launch configurations, and compilation interfaces (`utils/backend.py`,
  `utils/compile.py`, `utils/compile.sh`).

Determine which backends actually execute custom device-side code. The task catalog declares, per task, default
`backends` and opt-in `stretch_backends`; check that the declared backends match the fixtures that exist and the
backends that have run on a GPU.

### C. RL optimisation infrastructure

Inspect:

- `utils/grpo.py`
- `utils/reward.py`
- `utils/rollout_env.py`
- `utils/verification.py`
- `utils/config.py`
- `utils/stats.py`
- `utils/task_policy.py`, `utils/task_verification.py`, `utils/lora_verification.py` (the task-suite gates)
- `Architecture-decision/0001`–`0005` (CISPO preset, W&B tracking contract, CISPO GPU validation, trainer-metric
  definitions, backend diagnostic controls) and `RL.md`
- Any implementation of GRPO, DAPO, CISPO, or related policy optimisation objectives.

Do not assume an RL algorithm is implemented merely because it appears in a filename, configuration, document, or experiment name.

### D. Execution, validation, and benchmarking

Inspect:

- `utils/benchmark.py`
- `utils/compile.py`
- `utils/profiling.py`
- `utils/runtime.py`
- `utils/backend.py`
- `utils/provenance.py`
- `utils/task_suite.py`, `utils/task_catalog.py`, `utils/task_ceiling.py` (opt-in cuBLAS ceiling),
  `utils/gpu_lock.py`, `utils/diagnose.py`, `utils/diagnostic_review.py`, `utils/profile_artifacts.py`
- `cloud/modal_runtime.py`, `cloud/modal_tasks.py` (frozen-source task-suite runs), `cloud/modal_profile.py` and
  `cloud/modal_nsight_perf.py` (B200 profiling), `cloud/results_tools.py`
- `cloud/acceptance.py`
- `scripts/` (notably `validate.py`, `freeze_task_suite.py`, `query_nsys.py`, `profile_tasks.py`, `profile_b200.sh`)
- `tests/`
- `tracking/`
- `sweeps/`
- CI: `.github/workflows/agenticrl-cuda-kernels.yml` at the repository root (the Torch-free suite on Python 3.12)

### E. Documentation and experimental artifacts

Cross-reference:

- `README.md`
- `KNOWN_ISSUES.md`
- `VALIDATION.md`
- `Benchmark_diagnostics_task.md`
- `CISPO_GPU_batch_ranking.md`
- `TRITON_guide.md`
- `SKILL.md`
- Backend-specific skill documents
- Architecture decisions
- The task briefs (`*_task.md`), `EAGLE4_STAGE5.md`, `MULTI_LORA_STAGE6.md`, `GLOSSARY.md`,
  `TASK_SUITE_RESULTS.md`, `NSIGHT_PROFILING.md`, `NSIGHT_PERF_SDK.md`, `TORCH_PROFILER_PROFILING.md`
- The interactive reports linked from the README (Claude artifacts; they read fixed snapshots of committed CSVs, and
  viewing needs a claude.ai sign-in)

`HANDOFF*.md`, `TODO.md` and `INTERVIEW*.md` are gitignored working notes, not part of the repository.
- `artifacts/` benchmark JSON, logs, profiling output, generated candidates, and environment metadata.

Treat generated artifacts as historical experimental evidence, not automatically as canonical source code.

Check whether logged experiments correspond to the actual current source revision. Task-suite runs record a
per-file SHA-256 manifest (`source/task-source-manifest.json`) and report source mismatches; use it, and note that a
run frozen from an uncommitted tree records the previous commit as `git_revision` while its file hashes describe what ran.

---

## 2. Critical Priority: Audit Every `model.py` Implementation

This is the highest-priority task.

For every canonical model implementation, determine:

1. What mathematical operation is intended?
2. What operation does the actual code execute?
3. Is the implementation complete?
4. Does it invoke a genuinely custom GPU kernel?
5. Does it fall back to PyTorch, NumPy, or another reference implementation?
6. Are CUDA or DSL kernels compiled and launched correctly?
7. Are tensor shapes, strides, layouts, and dtypes handled correctly?
8. Are device placement and memory allocations handled safely?
9. Is the computation numerically correct?
10. Is autograd supported, unsupported, or unnecessary for the intended task?
11. Is the implementation deterministic where required?
12. Is it actually exercised by the tests and benchmark runner?
13. Can the kernel be optimised or replaced without breaking the public interface?
14. Is the submitted candidate identical to the kernel that was benchmarked?

### Required model audit matrix

Produce one row per model file:

| File | Intended operation | Actual implementation | Backend | GPU kernel invoked? | Correctness tests | Benchmark coverage | Status |
|---|---|---|---|---|---|---|---|

Use precise implementation statuses:

- Fully implemented and validated
- Implemented but incompletely validated
- Partially implemented
- Wrapper only
- Placeholder/stub
- Reference implementation only
- Broken
- Not implemented
- Unable to determine

Do not equate successful import, compilation, or symbol discovery with successful GPU kernel execution.

Inspect the actual call graph from model entrypoint to device-side computation.

For every defect, provide the exact file, function, relevant line numbers, supporting evidence, root cause, and proposed correction.

---

## 3. Critical Gap: Matrix Multiplication / GEMM

Investigate the existing matrix multiplication claims and implementation status.

### Current claims to verify

An earlier version of this brief started from the finding that the project had no completed GEMM kernel and only
element-wise AXPBY examples. That finding is out of date: the root example is still AXPBY, but the README and catalog
now claim these GEMM-family tasks:

- `gemm_div_sum_scale` (Stage 0): GEMM with a fused divide-sum-scale epilogue, Triton and CUDA.
- `gemm_tensor_core` (Stage 2): `A @ B + bias` on fp16/bf16 with fp32 accumulation; Triton (`tl.dot`), CUDA
  (WMMA/`mma.sync`), CuTe DSL (MMA atoms), and cuTile (`ct.mma`) as an opt-in stretch backend. The README claims the
  Triton fixture reaches 85.4% of cuBLAS at 4096³ bf16 on the H200 against a 60% gate, and CuTe DSL is correct at 2.6%.
- `grouped_gemm` (Triton only), `moe_dense_gemm` (dense all-expert FC1/FC2), and the attention and vocabulary-scoring
  kernels that use `tl.dot` (`flash_attention_fwd`, the tuned Stage 5 tree attention and candidate top-k).
- `fp8_gemm` and `int8_gemm`: contract only, no implementation.
- cuBLAS reference timings come from `utils/task_ceiling.py`, an opt-in diagnostic that times cuBLAS through PyTorch
  on the same inputs. No CUTLASS-profiler or cuDNN timings are claimed.

The framework must not imply that a custom matrix multiplication kernel or competitive GEMM benchmark exists without
direct code and benchmark evidence. **Treat each claim above as a hypothesis to verify against the code and the cited
run records, not as an established result.**

Search the entire codebase for:

- GEMM and matrix multiplication implementations
- `matmul`, `mm`, `bmm`, and `addmm`
- `torch.matmul`, `torch.mm`
- cuBLAS/cuBLASLt
- CUTLASS
- cuDNN
- `tl.dot`, `tl.dot_scaled`
- CuTe tiled MMA constructs
- Tensor Core, MMA, WMMA, and WGMMA operations
- TileLang GEMM implementations
- Custom CUDA GEMM implementations
- GEMM benchmark configurations and experimental results

Determine whether the repository has:

1. A genuine custom GEMM kernel.
2. Only framework-level matrix multiplication.
3. A placeholder or deferred GEMM design.
4. A functional benchmark comparing GEMM backends.
5. Verified cuBLAS, CUTLASS, or cuDNN reference timings.

Explicitly distinguish libraries that are relevant baselines from libraries that are actually used.

If GEMM remains unimplemented, mark it as **NOT IMPLEMENTED**, identify the missing files and interfaces, and recommend a minimal implementation path.

### Recommended GEMM development progression

Propose an implementation sequence covering:

**Level 1 — Correctness baseline**

- Naive CUDA GEMM
- PyTorch FP32 reference
- Shape and transpose validation
- Relative and absolute error metrics

**Level 2 — Tiled GPU execution**

- Shared-memory CUDA tiling
- Coalesced loads
- Register blocking
- Triton `tl.dot`
- Boundary masks
- FP16/BF16 input handling with appropriate accumulation precision

**Level 3 — Hardware-aware optimisation**

- Tensor Core utilisation
- Asynchronous data movement where supported
- Double buffering
- Warp-specialised execution where appropriate
- CuTe DSL / TileLang variants
- cuBLAS and CUTLASS comparisons
- Roofline analysis

**Level 4 — RL workload relevance**

- GEMMs encountered in policy forward/backward passes
- Attention projection GEMMs
- MLP projections
- Shape distributions encountered during RL fine-tuning
- Repeated small-batch and variable-sequence-length execution

Do not implement all levels immediately. Rank them according to current project maturity, expected performance impact, and research value. Levels 1–3 are claimed to exist in part; state which parts the evidence supports, and focus recommendations on what remains (for example the CuTe DSL GEMM's low efficiency, the unlaunched cuTile 60%-of-cuBLAS acceptance run, and Level 4's RL-shaped GEMMs).

---

## 4. CUDA, Triton, and Alternative Backend Correctness

For each backend, inspect:

- Kernel signature and interface
- Launch grid
- Block size and warp count
- Shared-memory consumption
- Register pressure
- Tensor layout assumptions
- Memory coalescing
- Memory alignment
- Bounds checking and masking
- Race conditions
- Synchronisation requirements
- Numerical precision
- Device compatibility
- Stream semantics
- Compilation and JIT caching
- Error propagation
- Fallback behaviour

Verify that every purported backend uses its intended compilation and execution mechanism.

Identify misleading cases where backend-specific wrappers ultimately execute ordinary PyTorch operations.

Check whether any backend-specific implementation is:

- An alias of another backend
- A reference-only implementation
- An untested experimental candidate
- Dependent on undocumented environment variables
- Dependent on unsupported hardware
- Missing runtime dispatch
- Silently falling back to CPU
- Incorrectly identified in telemetry or benchmark metadata

Require evidence that the intended device kernel was launched.

---

## 5. RL Algorithms: GRPO, DAPO, and CISPO

Audit the mathematical and implementation correctness of all RL optimisation components.

Known state to verify: the environment computes rewards (`utils/reward.py::compute_reward`) and group advantages
(`utils/grpo.py::build_batch`), with GRPO, DAPO and CISPO presets in `utils/config.py`. Clip bounds and the KL
coefficient are marked trainer-owned, and no policy loss, backward pass or optimiser exists in the repository. The
CISPO preset's upper clip (`clip_high=4.0`, a weight cap of 5.0) is taken from ms-swift's CISPO example, not from the
MiniMax-M1 paper, which gives no value. Separately, some of this math exists as *kernel tasks* (`grpo_group_norm`,
`cispo_clip_weight`, `log_softmax_gather`, `packed_varlen_reduction`), which are benchmarks for candidates, not code
used by a training loop.

### GRPO

Verify:

- Group-relative advantage computation
- Reward grouping and normalisation
- Per-token versus per-sequence weighting
- Policy/reference log-probability handling
- Importance sampling ratios
- Clipped surrogate objective
- KL regularisation, if applicable
- Response masking
- Variable-length sequences
- Numerical stability
- Zero-variance reward groups
- Gradient propagation and detach semantics

### DAPO

Determine whether DAPO is implemented, partially implemented, or only described.

Inspect applicable features such as:

- Dynamic sampling
- Clip-higher or asymmetric clipping
- Token-level policy gradients
- Overlong-response handling
- Variable-length rollout weighting
- Reward filtering and normalisation

Do not mark a feature missing solely because it differs from one particular DAPO implementation; distinguish required algorithmic properties from optional variants.

### CISPO

Verify the precise objective and its mathematical provenance.

Inspect:

- Importance sampling coefficient computation
- Clipping bounds
- Stop-gradient placement
- Token-level weighting
- Policy gradient correctness
- Handling of stale rollout policies
- Training stability
- Group ranking and reward aggregation

Where possible, derive the implemented loss from the source code and compare it with the intended mathematical formulation.

For each algorithm, provide:

- Expected objective
- Observed code-level objective
- Differences
- Whether the difference is intentional
- Severity
- Minimal correction
- Unit-test design

A named loss function is not sufficient evidence of algorithmic completeness.

---

## 6. End-to-End Agentic RL Training Integration

Determine whether the framework supports a complete and reproducible optimisation loop:

`Prompt → Policy rollout → Reward/verifier → Advantage estimation → Policy loss → Backpropagation → Optimiser step → Updated policy → Next rollout`

Identify which stages are actually implemented, mocked, partially connected, or missing.

Investigate whether custom GPU kernels contribute to:

- Training computation
- Advantage estimation
- Reward aggregation
- Masked reductions
- Log-probability computation
- Importance-ratio computation
- Policy-gradient losses
- Optimiser updates

Explicitly distinguish real GPU kernel execution from higher-level Python orchestration.

Investigate whether the framework is genuinely performing model fine-tuning or instead optimising/ranking kernel candidate programs through an RL-inspired search workflow.

These are different computational systems and must not be conflated.

Produce an end-to-end dependency graph identifying every incomplete execution path. ADR 0004 defines the trainer
metrics and names the first missing dependency (an episode ledger of generated tokens, behaviour log-probabilities and
turn counts); check whether anything beyond it exists.

---

## 7. Benchmarking, GPU Profiling, and Performance Claims

Audit the complete benchmarking pipeline.

Check:

- Correct warm-up methodology
- Compilation versus execution timing
- CUDA event timing
- Synchronisation correctness
- GPU clock and thermal effects
- Cold versus warm execution
- Kernel-only versus end-to-end latency
- Allocation and transfer overhead
- Measurement repetitions
- Variance and confidence intervals
- Backend-specific compilation overhead
- Shape, stride, and dtype coverage
- Reproducible hardware/software metadata
- Median, p95, and p99 latency
- Throughput
- Memory bandwidth
- Arithmetic intensity
- Achieved FLOP/s where applicable
- Performance relative to appropriate baselines

Inspect the committed benchmark logs and JSON artifacts. The reward measures device time with `torch.profiler`
(`utils/profiling.py`: device-side CUDA events only, two empty priming captures, per-event samples in a sidecar file).
`utils/benchmark.py` is an opt-in diagnostic layer (rotated inputs, cold-start timing, CUDA-graph replay) that the
reward must never read; `tests/test_cuda_graph_guardrails.py` claims to enforce that. Nsight Systems captures are
cross-checked against SQLite totals (`diag/nsys_stats_matches_sqlite`). NVIDIA's Nsight Perf SDK does not run on
Modal's B200 hosts (`NSIGHT_PERF_SDK.md`, run `nsight-perf-service-20261115T045206Z`). Runs exist on both an H200 and
a B200, so check any cross-GPU comparison for matched shapes, toolchains and drivers.

Determine whether each claim is supported by an exact benchmark record.

In particular, flag:

- Claimed GEMM speedups without GEMM execution
- Claimed cuBLAS/CUTLASS comparisons without measurements
- AXPBY results presented as matrix multiplication results
- Compilation success presented as runtime correctness
- GPU availability presented as verified kernel execution
- Unvalidated candidate rankings
- Stale or overwritten benchmark outputs
- Measurements without provenance
- Inconsistent benchmark shapes or precision settings
- Unsupported cross-GPU comparisons

For each existing benchmark, state whether the result is valid, questionable, invalid, or unverifiable.

Do not invent benchmark values or speedups.

---

## 8. Candidate Ranking and Multi-Agent Optimisation

Inspect how kernel candidates are generated, evaluated, scored, ranked, and promoted.

Verify:

- Candidate isolation
- Reproducible evaluation
- Correctness gating before performance scoring
- Penalisation of compilation failures
- Handling of execution timeouts
- Duplicate candidate detection
- Baseline-relative reward computation
- Normalisation of rewards
- Group construction
- Ranking consistency
- Statistical uncertainty
- Separation between training and held-out evaluation shapes
- Prevention of benchmark overfitting
- Candidate provenance
- Promotion of verified candidates only

Investigate potential reward hacking, including:

- Candidate code that returns hardcoded outputs
- Candidate code that bypasses the intended custom kernel
- Candidate code that manipulates benchmark timing
- Candidate code that dispatches to prohibited libraries
- Candidate code that passes a narrow correctness test but fails general workloads
- Candidate code that exploits validation harness weaknesses

Identify which attack surfaces are actually possible under the existing execution model. Defences to test rather
than assume: the provenance gate (task policy `authored-only-no-library-v1`, authored share ≥ 0.9), the per-task
`hack` and `wrong` controls, exact comparison of integer outputs, the `gpu_busy` exclusion from advantages, frozen-source
manifests, and the exploits `KNOWN_ISSUES.md` lists as not yet covered. "Multi-agent" in this repository's history
refers to the development workflow (Codex and Claude); check whether any multi-agent component exists at run time.

Recommend deterministic validators for these cases rather than relying exclusively on language-model grading.

---

## 9. Tests, CI, Reproducibility, and Cloud Execution

Inspect the full test suite and cloud runner.

Identify missing:

- Unit tests
- GPU integration tests
- Numerical gradient tests, where applicable
- Noncontiguous tensor tests
- Odd-dimension and tail-mask tests
- Backend parity tests
- Precision stress tests
- Device compatibility tests
- Performance regression tests
- Failure injection tests
- Deterministic replay
- Dependency pinning
- Hardware metadata
- CUDA compiler metadata
- Automated CI checks
- Reproducible Modal GPU jobs

For each important implementation, map source files to the tests that exercise them.

Distinguish tests that have merely been written from tests that have demonstrably passed in a relevant environment.

Where GPU access is unavailable, conduct static checks and classify runtime validation as pending. The process
tests need POSIX process groups, so run the suite on Linux or WSL (`python3 -m unittest discover`), not native Windows.
GPU validation runs only on Modal (H200 by default, B200 for Blackwell), is paid, and needs the user's approval.

---

## 10. Documentation and Research Integrity

Compare every substantive technical claim in the README and related documentation against the current implementation.

Classify claims as:

- Verified
- Partially supported
- Planned
- Deferred
- Contradicted by source
- Unsupported by benchmark evidence

Pay particular attention to performance claims, backend support, implemented kernel categories, RL algorithm completeness, and benchmark methodology.

For each misleading statement, provide replacement wording that accurately represents the codebase's current maturity.

Do not exaggerate research readiness or GPU performance.

---

## 11. Required Deliverables

Produce the following report.

### Deliverable A — Executive technical assessment

Provide:

- Overall implementation maturity
- Main architectural strengths
- Critical correctness risks
- Missing or incomplete features
- Research reproducibility risks
- Benchmark credibility assessment
- Readiness for public research release

### Deliverable B — Complete implementation inventory

List every substantive kernel, model entrypoint, RL objective, execution component, and benchmark.

Mark the current implementation and validation status.

### Deliverable C — File-by-file defect register

For every material issue, report:

| ID | Severity | File:line | Component | Observed defect | Evidence | Root cause | Proposed correction |
|---|---|---|---|---|---|---|---|

Severity:

- P0: Critical correctness, security, or benchmark-integrity failure
- P1: Major missing implementation or broken execution path
- P2: Partial implementation or substantial validation gap
- P3: Maintainability, documentation, or optimisation opportunity

Do not manufacture issues to fill severity categories.

### Deliverable D — `model.py` deep audit

Provide a separate detailed subsection for every canonical `model*.py` file.

Include:

- Actual call graph
- Implemented operation
- Device execution mechanism
- Correctness status
- Backend implementation status
- Missing functionality
- Associated tests
- Recommended changes

### Deliverable E — GEMM implementation gap analysis

State whether any real custom GEMM exists.

If not, explain exactly what is missing and propose a phased development strategy, starting from a validated CUDA/Triton implementation and progressing to hardware-optimised kernels.

### Deliverable F — RL objective correctness assessment

Compare the actual code with the expected GRPO, DAPO, and CISPO formulations.

Provide explicit equations and highlight any discrepancies.

### Deliverable G — Ranked remediation roadmap

Produce an engineering roadmap prioritised by:

1. Correctness
2. Functional completeness
3. Benchmark validity
4. GPU performance
5. End-to-end RL integration
6. Research reproducibility
7. Documentation quality

For each proposed task, include estimated engineering difficulty, dependencies, proposed files, and objective acceptance criteria.

### Deliverable H — Research-readiness verdict

Evaluate whether the repository currently supports:

- Reproducible GPU kernel benchmarks
- Meaningful backend-to-backend comparisons
- Genuine performance claims
- Scientifically valid RL-based kernel optimisation
- Publication-quality empirical evaluation

Distinguish what has already been demonstrated from what remains a research objective.

---

## 12. Evidence and Execution Requirements

Follow these rules throughout the audit.

1. Read source code rather than relying on documentation summaries.
2. Cite exact GitHub file paths and line numbers for each finding.
3. Distinguish confirmed defects from hypotheses and untested concerns.
4. Trace implementations through their actual execution paths.
5. Do not assume that compilation proves runtime correctness.
6. Do not assume that a benchmark artifact proves its associated source was executed.
7. Do not invent measurements, kernel launches, test results, or experimental outcomes.
8. Execute available tests only when the environment permits and report exact commands and outcomes. Do not launch paid Modal GPU runs without the user's approval.
9. Preserve the existing codebase; do not silently implement fixes during the audit.
10. Do not substitute generic best practices for repository-specific evidence.
11. If a component is inaccessible or cannot be evaluated, explicitly mark it as unverified.
12. Cite the inspected commit SHA so the report can be reproduced, and cite run IDs for every GPU number.

**Final objective:** Establish a technically defensible picture of the repository's actual implementation state, particularly the `model*.py` files, custom GPU kernels, GEMM functionality, RL optimisation correctness, and benchmark credibility.

The resulting report should be actionable for a research engineer preparing this framework for rigorous GPU performance experimentation and an eventual peer-reviewed systems or reinforcement learning publication.
