# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A provenance-gated RL environment for GPU kernel generation: a policy writes a `model_new.py` (CUDA C++, Triton, TileLang, cuTile or CuTe DSL), and the harness compiles, verifies and profiles it, credits only device time in agent-authored kernels, and turns that into a GRPO/DAPO/CISPO reward. Read `AGENTS.md`, `README.md`, `VALIDATION.md`, `KNOWN_ISSUES.md` and the **Ground rules** at the top of `Benchmark_diagnostics_task.md` before changing behaviour. `GLOSSARY.md` defines the jargon (BGMV/SGMV, provenance, relative bound, graph replay).

## Commands

Linux/WSL2 only: `run_task` uses POSIX process groups, so process tests fail on native Windows. From a Windows checkout, run everything through `wsl.exe -e bash -lc "cd /mnt/c/.../AgenticRL-CUDA-kernels && ..."`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover            # Torch-free suite (what CI runs)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover   # CPU Torch + Triton interpreter; slow
python3 -m unittest tests.test_suite_log.SuiteLogTests.test_payload_drops_logs_and_keeps_provenance  # one test
.venv/bin/python -m scripts.validate --device cpu --timeout-s 600  # CPU correctness report
.venv/bin/python -m scripts.validate --suite stage6 --device cuda --case-set smoke --fixtures all --backends cuda,triton  # task suite (GPU)
python -m utils.rollout_env <workdir> --algo grpo --turns-used 1   # one scored rollout
```

CI: `.github/workflows/agenticrl-cuda-kernels.yml` runs the Torch-free suite on Python 3.12 with pyyaml. There is no formatter or linter config.

GPU work runs on Modal (H200 by default via `MODAL_GPU`), always with `--detach`:

```bash
.venv/bin/python -m scripts.freeze_task_suite --tasks stage6 --output <frozen dir>
cd <frozen dir> && PYTHONPATH=$PWD MODAL_GPU=H200 <repo>/.venv/bin/modal run --detach cloud/modal_tasks.py \
  --tasks stage6 --case-set acceptance --backends cuda,triton --fixtures all --cuda-graphs --timeout-s 1200 --run-id <id>
.venv/bin/modal volume get agenticrl-cuda-artifacts /<run-id>/ <local dir>   # outputs land in this volume
.venv/bin/python -m tracking.suite_log artifacts/runs/<run-id> [--dry-run]    # log a suite run to W&B + read back
```

## Architecture

**Rollout pipeline** (`utils/rollout_env.py::run_task`): `utils/backend.py` detects the backend by AST/regex scan (no import), then compile, verification and profiling run as separate child processes under one wall-clock deadline. Profiling classifies device time into agent/library/torch/memory (`utils/provenance.py`); `utils/reward.py::compute_reward` maps verify + timings + provenance to the milestone reward (−1/1/2/3); `utils/grpo.py::build_batch` computes group advantages. `run_task` returns a flat, JSON-safe dict whose key set is pinned by `tests/support.py::DEFAULT_RESULT_KEYS` and documented in `Architecture-decision/0002-wandb-tracking-contract.md`.

**Config** (`utils/config.py::RolloutConfig`): every field automatically gets a `--kebab-case` CLI flag (tested); presets (GRPO/DAPO/CISPO) apply before overrides. `cuda_graphs` requires `benchmark`.

**Opt-in measurement layers**, never reward inputs: `utils/benchmark.py` (rotated benchmark, `CapturedGraph` replay that owns its static buffers), `utils/profile_artifacts.py` (diagnostic pass, profiler sidecar files), and `utils/gpu_lock.py` (per-GPU-UUID `flock` held across profiling, benchmark and diagnostics; waiting extends the deadline; a capped wait yields reason `gpu_busy`, which `build_batch` excludes from advantages). `tests/test_cuda_graph_guardrails.py` asserts the reward code never reads `bench`/`graph` values.

**Task suite** (stages 0–6, ~35 tasks in `tasks/<name>/`): shapes and dtypes come from `utils/task_catalog.py` (`select_cases(task, "smoke"|"acceptance"|"stratified")`) so a candidate cannot redefine them. Each task folder holds a trusted `model.py` reference plus fixtures: `model_new_{cuda,triton}.py` (honest), `_hack` (Torch-op; must fail provenance), `_wrong` (must fail verification), and for Stage 6 `_wrong_padding`, `_wrong_skip` and a diagnostic-only `_fp16_intermediate`. `utils/task_suite.py::evaluate_case` builds a workdir per (task, case, backend, fixture), applies task-only gates (`utils/task_policy.py`: provenance `authored-only-no-library-v1`, authored share ≥ 0.9; `utils/task_verification.py`: exact integer outputs; `utils/lora_verification.py`: Stage 6 relative-error bound) and benchmarks only honest fixtures. `scripts/validate.py --suite` drives it; `cloud/modal_tasks.py` forwards it to Modal from a frozen source (`scripts/freeze_task_suite.py` writes `source/task-source-manifest.json`, and the run reports source mismatches).

Stage-specific references live in `utils/eagle_*.py` (Stage 5, EAGLE-4 speculative decoding) and `utils/lora_*.py` (Stage 6, multi-LoRA BGMV/SGMV); see `EAGLE4_STAGE5.md` and `MULTI_LORA_STAGE6.md`.

## Hard constraints

- Protected upstream files are checksummed in `artifacts/upstream.json` and must stay byte-identical: `model.py`, `model_new.py`, `binding.cpp`, `binding_registry.h`, `kernels/axpby.cu`, `kernels/axpby_binding.cpp`, `SKILL.md`. The profiling stdout line and the "`--json` is the last stdout line" rule are also pinned.
- `utils/config.py`, `reward.py`, `grpo.py`, `rollout_env.py` (and other parent modules) must import under `python -S` without Torch; import GPU packages lazily. W&B/Weave imports stay in `tracking/`.
- Add result keys, never rename or redefine them. Never fabricate numbers: an unmeasured value is `null` with a reason, and every documented number cites its run ID and command.
- No new Python dependencies without the user's approval.
- Ask the user before changing reward semantics, timing meanings, verification tolerances or protected files, before paid Modal GPU runs, and before restarting a GPU job that was killed.

## Repo workflow

- Codex works in WSL (`.git` read-only there); Claude reviews, tests (in WSL), groups and commits from the Windows checkout (`core.autocrlf`, CRLF). Commit with focused imperative subjects ("Add …", "Fix …", "Record …"). Push only when asked.
- `HANDOFF*.md` files are gitignored working notes; never commit them.
- GPU evidence goes in `artifacts/runs/<run-id>/`: JSON/JSONL/logs/Markdown plus `source/task-source-manifest.json` only (no caches, build output or source copies; `.gitignore` drops suite scaffolds). Scan it for secrets and the user's email before committing. Leave `artifacts/volume-20261108T014026Z/` and `artifacts/wandb-offline-smoke-20261108T013827Z/` untracked.
- W&B credentials come from `.env` (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`); load them without printing. MLflow is out of scope.
