# CUDA-Agent environment extension

This folder extends the example environment from [CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent), pinned in [artifacts/upstream.json](artifacts/upstream.json). Its `agent_workdir/` files live directly here. The reference model, fixed bindings, stock CUDA kernels and stock `model_new.py` are byte-identical to that revision. All new work, runtimes and validation artifacts stay in this folder.

Python 3.10+ is required. Install `requirements.txt` in a local virtual environment for GPU use; CUDA also needs the upstream build tools, nvcc and Ninja, on PATH. No trainer, policy loop, Modal, MLflow or W&B integration is included in this phase.

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"
bash utils/compile.sh && python3 -m utils.verification && python3 -m utils.profiling
python -m utils.rollout_env . --algo grpo --turns-used 1
```

For CPU testing, install Torch from `https://download.pytorch.org/whl/cpu`, then Triton and NumPy (the interpreter's NumPy prerequisite). Run `python -m scripts.validate --device cpu --timeout-s 600` to create separate task fixtures without replacing the stock CUDA model. Run `python -m unittest discover -v` for the complete suite; GPU assertions skip explicitly without CUDA/nvcc. Raw evidence and limitations are in [VALIDATION.md](VALIDATION.md) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Triton backend + provenance-gated GRPO/DAPO reward

Use `model_new_triton.py` as a task's `model_new.py`; see [SKILL_TRITON.md](SKILL_TRITON.md). `--backend auto|cuda|triton` selects compilation. Profiling preserves the upstream timing line and appends per-kernel times and provenance with `--json`; unknown names count as Torch, and memcpy/memset is excluded from the gate denominator.
GPU measurements below are **not measured** on this machine; CPU runs earn no performance reward. Reproduce the case rows with `python -m scripts.validate --device cpu --timeout-s 600` (raw JSON in `artifacts/results.json`); GPU proof uses `python -m unittest -v tests.test_provenance.GPUIntegrationTests`.

| Case | Eager µs | Compile µs | CUDA µs | Triton µs | Provenance % | Outcome |
|---|---|---|---|---|---|---|
| Honest CUDA | not measured | not measured | not measured | — | not measured | GPU test skipped |
| Honest Triton | not measured | not measured | — | not measured | not measured | CPU correctness; unmeasured reward |
| Torch hack | not measured | not measured | — | not measured | not measured | CPU correctness; GPU gate test skipped |
| Timeout | not measured | not measured | — | not measured | not measured | timeout |
| Max turns | not measured | not measured | — | not measured | not measured | max_turns |
| Overlong (DAPO) | not measured | not measured | — | not measured | not measured | shaping unit-tested; GPU check skipped |

Synthetic honest/honest/hack rewards (3, 3, -1) and advantages below come from `python -m scripts.validate --device cpu --timeout-s 600`; they are arithmetic fixtures, not GPU results. Overlong shaping leaves the milestone base unchanged.

| Preset | Triton advantage | CUDA advantage | Hack advantage | Tied group |
|---|---|---|---|---|
| GRPO (synthetic) | 0.707106 | 0.707106 | -1.414213 | kept, zero advantage |
| DAPO (synthetic) | 0.707106 | 0.707106 | -1.414213 | dropped |

**Phase 2: Modal rollouts.** Call `utils.rollout_env.run_task(workdir, cfg, turns_used)` with `RolloutConfig.from_args(...)`; start with `--max-turns 20 --max-context-tokens 32768 --timeout-s 180 --group-size 8`. The future loop enforces turns/context; the environment also checks the turn count and enforces the shared wall-clock deadline. CPU or absent GPU measurements return null fields and reason `unmeasured`, never positive rewards.

**Phase 3: GRPO vs DAPO.** Four independent toggles: clip range and sequence/token loss (trainer-owned), tied-group filtering and hard/soft overlong shaping (environment-owned). Presets apply before CLI overrides. W&B namespaces are `env/`, `reward/`, `adv/`, `turns/`, `meta/`; omit `log/*` before logging. `RolloutConfig.to_dict()` is the MLflow parameter set, with tags as a JSON-safe list. No tracking-library imports are required.
