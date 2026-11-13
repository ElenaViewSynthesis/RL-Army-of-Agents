# Repository Guidelines

## Project Structure & Module Organization

`kernels/` contains CUDA, Triton, TileLang, cuTile and CuTe DSL implementations. `utils/` owns backend discovery, compilation, verification, profiling, rewards and rollout configuration. `tests/` contains unittest suites and fixtures. `scripts/` produces validation reports; `cloud/` runs Modal jobs. Optional tracking lives in `tracking/`; evidence belongs in `artifacts/`, and design decisions in `Architecture-decision/`.

Before editing, read `README.md`, `VALIDATION.md`, `KNOWN_ISSUES.md` and the binding Ground rules in `Benchmark_diagnostics_task.md`. Consult the relevant `SKILL_<BACKEND>.md` for kernel policy.

## Build, Test, and Development Commands

Run commands from this directory on Linux or WSL2:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 -m unittest discover -v
.venv/bin/python -m unittest discover -v
```

The two suites cover dependency-light checks and installed-runtime paths. CPU Triton interpretation additionally needs NumPy. For a CPU correctness report:

```bash
.venv/bin/python -m scripts.validate --device cpu --timeout-s 600
```

On a configured GPU host, `bash utils/compile.sh`, `python -m utils.verification` and `python -m utils.profiling --json` run the stock pipeline. The pinned cloud environment uses `requirements-gpu.txt`; `modal run --detach cloud/modal_runtime.py --tests` runs full GPU validation.

## Coding Style & Naming Conventions

Use four-space Python indentation, snake_case functions/modules and descriptive class names. Match neighboring formatting; no project formatter configuration is present. Extend `utils/backend.py::BACKENDS` for backend behavior. Authored kernel names must contain at least four characters and avoid generic names such as `kernel`, `main` and `add`.

Keep pure parent modules importable under `python -S` without Torch; import GPU packages lazily. Keep W&B/Weave imports in tracking code. Preserve flat JSON-safe results, timing meanings and reward rules; diagnostics never feed rewards.

## Testing Guidelines

Name tests `tests/test_*.py` using `unittest.TestCase`. Add regression coverage for changed behavior and explicit skips for missing GPUs/packages. Report raw test summaries and skip reasons. Record GPU commands, run IDs, device identity and versions; historical acceptance does not validate newer code. Never fabricate timings or relax unknown-event attribution.

## Commit & Pull Request Guidelines

Use focused commits with imperative subjects, following history: “Add …”, “Fix …”, “Record …”. PRs should explain behavior, link relevant issues, list validation commands/results and update affected documentation.

Seek approval before changing protected upstream files, reward semantics or tolerances; protected files are enumerated in the Ground rules. Never commit credentials or caches. Tracking is opt-in; retain intended validation evidence with run IDs.
