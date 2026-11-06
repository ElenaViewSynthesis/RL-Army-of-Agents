# Validation report

All commands below run inside `AgenticRL-CUDA-kernels`. No project or secret
files outside this folder were changed. The upstream environment was imported
from commit `473025c8af7878e928525138b5cb327d6bcdf9dd`; original SHA-256 checksums
are in [artifacts/upstream.json](artifacts/upstream.json). Tests assert that
model.py, binding.cpp, binding_registry.h, model_new.py and both stock kernel
files remain byte-identical.

## Commands and raw output

The original stock pipeline was attempted before editing the utilities:

```bash
bash utils/compile.sh && python3 -m utils.verification && python3 -m utils.profiling
```

Raw output: [artifacts/stock_before.log](artifacts/stock_before.log). It exited
at compilation with `ModuleNotFoundError: No module named 'torch'`; the shell
reported `[TIME] Compilation took 1.06s`. That is a failed baseline attempt,
not a GPU benchmark. There is no visible NVIDIA GPU or nvcc on this host.

Dependencies were installed entirely locally (uv selects Python 3.13 here):

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python 'torch>=2.3' --index-url https://download.pytorch.org/whl/cpu
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python 'triton>=3.0' numpy
```

NumPy is the Triton interpreter prerequisite. GPU runtime requirements remain
Torch and Triton only. The CPU installation is for local correctness checks;
replace the CPU Torch build in a GPU runtime.

The final stock pipeline uses that local interpreter:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 bash -o pipefail -c '{ bash utils/compile.sh && python3 -m utils.verification && python3 -m utils.profiling; } 2>&1 | tee artifacts/stock_after.log'
```

Raw output: [artifacts/stock_after.log](artifacts/stock_after.log). It reports
missing Ninja and `[TIME] Compilation took 42.02s`. CUDA/nvcc are also absent;
this cannot establish a successful CUDA regression run. A pure
test separately asserts the exact upstream `Torch Baseline: ...us, Torch
Compile: ...us, CUDA Extension: ...us` formatting.

```bash
python3 -m unittest discover -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.validate --device cpu --timeout-s 600
```

Raw outputs: [tests without dependencies](artifacts/tests_no_torch.log),
[complete local test run](artifacts/tests.log),
[actual rollout JSON lines](artifacts/rollouts.log), and
[structured evidence](artifacts/results.json). Runtime versions and missing
GPU/build-tool executables are recorded in [environment.json](artifacts/environment.json).

| Command | Raw summary |
|---|---|
| `.venv/bin/python -m unittest discover -v` | `Ran 41 tests in 612.805s`, `OK (skipped=5)`; all five skips require GPU/nvcc |
| `python3 -m unittest discover -v` (final pure-Python run) | `Ran 43 tests in 11.013s`, `OK (skipped=11)`; skips include the CPU interpreter checks because system Python has no Torch |
| `python3 -m unittest -v tests.test_rollout_env` | `Ran 11 tests in 16.481s`, `OK` ([raw](artifacts/process_tests.log)) |
| `python3 -m unittest -v tests.test_backend_provenance` | `Ran 9 tests in 2.026s`, `OK` ([raw](artifacts/backend_tests.log)) |

The final pure-Python run includes the added syntax-error and immediate CPU
profiling-skip regressions. Together these runs cover the current 43 checks:
32 pure-Python/process checks and six CPU/interpreter checks passed; five
GPU-only checks remain skipped. Runtime seconds above describe test/process
execution on this host, not GPU performance.

The initial CPU run used the unchanged default deadline and timed out after
180.05 seconds: its Triton smoke compile succeeded in 143.19 seconds, leaving
too little budget for verification. The exact run was
`.venv/bin/python -m scripts.validate --device cpu`; raw evidence is retained
in [artifacts/rollouts_cold_start.log](artifacts/rollouts_cold_start.log).
Final CPU validation explicitly uses `--timeout-s 600` for this host; the
environment's default remains 180. All these seconds are local process times,
not GPU kernel performance measurements.

## Results

The full eager/compile/CUDA/Triton timing table and the synthetic group
advantages are in [README.md](README.md#triton-backend--provenance-gated-grpodapo-reward).
All GPU times and provenance percentages are **not measured**. This table
comes from `.venv/bin/python -m scripts.validate --device cpu --timeout-s 600`;
each row's raw flat dictionary is in [results.json](artifacts/results.json).

| Fixture | CPU verification | Reward | Reason | Overlong penalty |
|---|---|---|---|---|
| Honest Triton | passed | -1 | unmeasured | 0 |
| Torch-op hack | passed | -1 | unmeasured | 0 |
| Timeout (5-second budget) | not completed | -1 | timeout | 0 |
| Max turns (25 used, cap 20) | skipped | -1 | max_turns | 0 |
| DAPO overlong (19 used) | passed | -1 | unmeasured | 0 |
| Honest CUDA | not run | not measured | GPU unavailable | not measured |

The CPU overlong case cannot earn a milestone before measurements exist,
so its failure is not shaped. The pure reward demo establishes shaping on
identical synthetic measurements, with an unchanged milestone base:

```bash
python3 - <<'PY'
import json
from utils.config import RolloutConfig
from utils.reward import compute_reward
for algo in ('grpo', 'dapo'):
    print(json.dumps({'algo': algo, **compute_reward(True, 100, 80, 50, 1.0, RolloutConfig.preset(algo), 19)}))
PY
```

Raw arithmetic output: [reward_demo.log](artifacts/reward_demo.log) (the
recorded version also labels every row as synthetic, not a GPU measurement).
GRPO returns base/reward 3 with penalty 0; DAPO returns base 3, penalty 0.75
and reward 2.25. The honest/honest/hack synthetic group has rewards 3, 3, -1,
and advantages 0.707106, 0.707106, -1.414213 under either preset (rounded).
Reproduce that group with `python -m scripts.validate --device cpu --timeout-s 600`;
the group size is explicitly 3 for this trio, not the default 8. Both presets
use population standard deviation plus epsilon; the tied-group test confirms
GRPO retains all-zero advantages while DAPO drops the tied group.

## What the tests establish

- Pure-Python config, reward, advantage and environment modules import with
  `python -S` and never import Torch. Every config field has a CLI flag;
  presets are applied before independent overrides.
- Reward boundaries, provenance failures, missing/invalid measurements,
  hard turn caps, linear shaping, tied-group filtering, same-task grouping,
  population standardisation, and stable flat metric keys are exercised.
- Real child processes demonstrate crash isolation, timeout termination of
  descendants, concurrent workdir isolation, and correct import resolution
  even for upstream-style namespace packages and a poisoned task utils/.
- CPU interpreter correctness covers the reference example and scalar,
  empty, short, non-divisible and matrix shapes, three floating dtypes and
  four scale values; inputs must remain unchanged.
- CPU `run_task` fixtures test honest Triton, a Torch-op hack, timeout, turn
  cutoff, wrong output and a crashing model. Missing device measurements
  remain null and cannot earn positive rewards.
- Synthetic profiler fixtures test authored-name attribution, CUDA-only
  library allowances, memory exclusion, unknown-event rejection and the
  Torch-hack gate. Mocked subprocess fixtures test end-to-end reward wiring;
  these are explicitly not hardware performance evidence.

GPU-only tests in `tests/test_provenance.py` require honest CUDA and Triton
to pass both correctness and provenance, require the Torch-op hack to pass
correctness and fail provenance, rank that hack last under both presets,
and check actual DAPO turn shaping. They remain skipped without hardware.
On a GPU runtime, run:

```bash
python -m unittest -v tests.test_provenance.GPUIntegrationTests
python -m scripts.validate --device cuda
```

Only that run can replace the README's **not measured** GPU timing and
provenance cells. No kernel event names were observed on real hardware here;
synthetic event labels are classifier fixtures, not a claimed profiler trace.

## Why this extends the supplied environment

- **Backend generality:** source discovery and the same compile/verify/profile
  contract support CUDA extensions and authored Triton JIT kernels.
- **Provenance gating:** device-time attribution adds an enforceable reward
  check for the specified ordinary Torch-op hack, beyond instruction text.
  Name matching is a heuristic; its limitations are documented.
- **Critic-free reward handling:** environment-owned tied-group filtering and
  turn shaping make the discrete milestone signal configurable for GRPO and
  DAPO without the paper's critic. Trainer-owned clipping/loss aggregation are
  exposed as configuration rather than implemented as an invented trainer.

## Step 1: benchmark mode

An opt-in `utils/benchmark.py` (plus `utils/stats.py` and a
`RolloutConfig.benchmark` stage in `run_task`) was added. With the flag off,
the profiling argv, the 18-key result set and the reward are byte-for-byte
unchanged (`git diff HEAD -- utils/profiling.py` is empty). All `bench/*`
keys are diagnostics: on this host only the CPU unavailable record and the
pure-Python statistics/rotation paths ran; every GPU latency, device key and
CUDA-graph result remains **not measured**.

```bash
python3 -m unittest discover -v
# Ran 57 tests in 9.284s
# OK (skipped=13)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest -v tests.test_benchmark tests.test_rollout_env.RolloutProtocolTests
# Ran 21 tests in 6.321s
# OK (skipped=2)

CUDA_AGENT_DEVICE=cpu TRITON_INTERPRET=1 .venv/bin/python   # direct run_task, benchmark=True, triton fixture
# "env/verify_ok": true, "reward/reason": "unmeasured",
# "bench/available": false, "bench/unavailable_reason": "CUDA_AGENT_DEVICE=cpu"
```

The 13 skips are GPU/CUDA or Torch-dependent checks (GPU provenance, GPU
benchmark/graph tests, Torch interpreter and Triton-kernel tests under system
Python). The CPU rollout exercised the real 4th stage subprocess and merged
only the two truthful `bench/*` unavailable keys; no timings were fabricated.
A benchmark stage that crashes, times out or emits malformed output degrades
to `bench/available: false` with a reason and never changes the reward.

## GPU runtime pin

The cloud-GPU runtime was pinned as `requirements-gpu.txt` (pip-installable;
uv's first-index strategy cannot resolve the `+cu130` wheel), a torch-free
preflight `utils/runtime.py`, and `cloud/modal_runtime.py` targeting
`nvidia/cuda:13.3.1-devel-ubuntu24.04` + Python 3.12 + `torch==2.14.1+cu130`
+ `cuda-tile==1.8.0`/`tileiras>=13.3,<13.4`. **No GPU or Modal execution was
performed** — all results below are local, torch-free checks and injected
fixtures; Modal API use was verified by importing the module only.

```bash
python3 -m unittest -v tests.test_runtime
# Ran 9 tests in 1.194s
# OK

UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python modal   # modal==1.6.3
.venv/bin/python -c "import importlib.util,sys; spec=importlib.util.spec_from_file_location('m','cloud/modal_runtime.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(type(m.image), type(m.app))"
# <class 'modal.image.Image'> <class 'modal.app.App'>

python3 -m unittest discover -v
# Ran 90 tests in 37.982s
# OK (skipped=31)
```

`python -m utils.runtime --json --require cutile` on this host exits 1 with an
unavailable reason, and `--json` prints a flat record with all `ok_*` False —
the preflight reports, never fabricates. `modal run` was never invoked.

The pin was then extended with `tilelang==0.1.15` and
`nvidia-cutlass-dsl[cu13]==4.8.1` (the `[cu13]` extra pulls the cu13 libs plus
`cuda-bindings`, which `kernels/cutedsl` imports), `runtime/ok_tilelang` and
`runtime/ok_cutedsl` preflight profiles, and
`scripts/record_kernel_names.py`, which stages compile + profiling per
backend like `run_task` and maps every `kernel_us/*` event through
`classify_kernel`. Cloud `gpu_tests` now runs the require gate (recorded, not
fatal), the stock pipeline, `record_kernel_names`, the GPU suites and
validate, and copies `kernel_names.json` to the artifacts volume.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest -v tests.test_runtime tests.test_record_kernel_names
# Ran 13 tests in 4.319s
# OK

.venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('m','cloud/modal_runtime.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"
# ok

python3 -S -c "from utils import runtime; import scripts.record_kernel_names; import sys; assert 'torch' not in sys.modules"
# (exit 0; torch-free)

python3 -m unittest discover -v
# Ran 94 tests in 21.066s
# OK (skipped=31)
```

`record_kernel_names` was verified against synthetic stage fixtures only; no
real GPU kernel names have been recorded yet — `artifacts/kernel_names.json`
is produced by the Modal or bare-host GPU run.

## DSL backends (TileLang, cuTile, CuTe DSL)

A backend registry (`utils/backend.py`) replaced the hard-coded CUDA/Triton
branches, and TileLang, cuTile and CuTe DSL backends were added on top of it.
CUDA and Triton behaviour is unchanged: after the refactor, the full suite and
the Torch-dependent backend, kernel, benchmark, config and rollout tests passed
without edits. No DSL package is installed on this host and there is no GPU,
so **no DSL kernel has run**. Only source discovery, backend detection and
classification are exercised here.

These runs used an export of the committed tree
(`git -c core.autocrlf=false archive HEAD`), so uncommitted work elsewhere in
the folder is excluded:

```bash
python3 -m unittest discover -v
# Ran 81 tests in 8.806s
# OK (skipped=31)

python3 -m unittest -v tests.test_backend_provenance.DSLBackendTests tests.test_dsl_kernels
# Ran 15 tests in 3.188s
# OK (skipped=9)
```

The 31 skips are the 13 earlier GPU/Torch-dependent checks plus 18 DSL tests:

- 9 kernel tests: shapes, tails, dtypes, input validation, and CUDA-graph
  replay on the current stream.
- 9 `run_task` integration tests: the honest kernel passes provenance, and the
  Torch-op hack fails it and ranks last.

Each skips with a reason naming the missing GPU or package (for example
`requires a CUDA GPU, Torch and tilelang`). On a GPU host with the extras
installed:

```bash
python -m unittest -v tests.test_dsl_kernels tests.test_provenance
python -m scripts.validate --device cuda
```

`scripts.validate` lists each DSL backend it could not run under
`unavailable_backends` in `results.json`, with the reason: a missing package,
or `GPU only (no CPU interpreter)` with `--device cpu`.

## Scope cuts and Modal implications

The optional fused Level-2 GEMM kernel was deferred in favour of the required
pipeline and validation. Actual RL, Modal, agent-loop, MLflow and W&B code
remain deferred exactly as specified. `run_task` is callable from a provisioned
task workdir without depending on the caller's cwd or importing Torch in the
parent process. It inherits CUDA_VISIBLE_DEVICES and uses task-local caches.

Cold nvcc/JIT/autotune and torch.compile warmup count against the same
180-second default deadline. Level-3 tasks may need a larger timeout; a warm
dependency image reduces cold-start cost. Avoid concurrent timing on a shared
GPU: contention changes rewards. A production Modal sandbox must enforce
filesystem/network/resource restrictions and a container-level deadline for
driver hangs or processes that escape the ordinary process group. See
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) for attribution and experiment-design limits.
