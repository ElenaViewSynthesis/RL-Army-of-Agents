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
and check actual DAPO turn shaping; its DSL classes and `tests/test_dsl_kernels.py`
do the same for TileLang, cuTile and CuTe DSL, and
`tests/test_benchmark.GPUBenchmarkTests` checks the opt-in benchmark stage.
They remain skipped without hardware. On a GPU runtime, run (in this order):

```bash
pip install -r requirements-gpu.txt   # pinned stack incl. tilelang and nvidia-cutlass-dsl[cu13]
python -m utils.runtime --json --require cuda,triton,tilelang,cutile,cutedsl
bash utils/compile.sh && python -m utils.verification && python -m utils.profiling --json
python -m scripts.record_kernel_names --device cuda   # -> artifacts/kernel_names.json
python -m unittest -v tests.test_provenance tests.test_dsl_kernels tests.test_benchmark.GPUBenchmarkTests
python -m scripts.validate --device cuda              # -> artifacts/results.json
```

`modal run cloud/modal_runtime.py --tests` runs the same sequence on Modal and
copies `kernel_names.json`, `results.json` and `environment.json` to the
`agenticrl-cuda-artifacts` Volume; `modal run cloud/modal_runtime.py` alone is
the cheap preflight.

Only that run can replace the README's **not measured** GPU timing and
provenance cells. No kernel event names were observed on real hardware here;
synthetic event labels are classifier fixtures, not a claimed profiler trace.
Read `kernel_names.json` (observed names, classifier verdicts and
`unmatched_agent_names` per backend) before interpreting any provenance result.

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

## First Modal GPU validation (2026-11-06)

Initial HEAD: `3b8a0d7`. The pre-existing GPU command-block edit above was
preserved. The checkout is part of the parent CUDA-RL repository. Focused
commits cannot be created because `.git` is mounted read-only; the attempted
commit failed creating `.git/index.lock`. Nothing was pushed.

Commands:

```bash
.venv/bin/modal run cloud/modal_runtime.py
.venv/bin/modal run cloud/modal_runtime.py --tests
```

The first image build failed resolving `cuda-toolkit>=13.3,<13.4` against
Torch's `cuda-toolkit==13.0.3` runtime dependency. Raw build output:
[failed image build](artifacts/modal-build-im-u7ElhT76HK3e4MiGfUc5JZ.log).
Direct pins for the 13.3.1 compiler components resolved the conflict.
The combined image subsequently built successfully, and preflight run
`20261106T142735Z` passed ([full build/preflight log](artifacts/modal-preflight-20261106T142735Z.log)).

The initial full GPU run `20261106T142830Z` found a separate CUDA build defect:
ATen requires C++20, but the harness forced C++17. Removing the override lets
Torch select its required standard. Protected upstream sources are unchanged.
This first run's [results](artifacts/results-20261106T142830Z.json) and
[names](artifacts/kernel_names-20261106T142830Z.json) are retained, including
the failed CUDA fixture. The GPU suite's raw summary was:

```text
Ran 30 tests in 403.421s
FAILED (failures=2)
```

Both failures concern the failed CUDA rollout (honest CUDA and group ranking).
All three DSL kernel suites and their honest/hack provenance tests passed.
The subsequent stock CUDA pipeline in run `20261106T185236Z` passed all five
verification checks and recorded the authored CUDA event.

Local commands and raw summaries:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v
Ran 94 tests in 52.212s
OK (skipped=31)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
Ran 94 tests in 557.828s
OK (skipped=25)

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v  # after build fix
Ran 94 tests in 66.922s
OK (skipped=31)
```

The cloud runner's uploaded `environment.json` was historical CPU data.
`environment-uploaded-stale-<run_id>.json` preserves it. Corrected environment
records are reconstructed from each run's fresh `require.stdout.log`, with
that source explicitly recorded. Future runs now write environment.json from
the runtime probe directly; that packaging change awaits remote execution.

Name matching remains a heuristic, not proof of authorship. The classifier
change is awaiting explicit user approval. Unknown events still count as Torch.

### Successful H200 rerun 20261106T185236Z

Command: `python -m scripts.validate --device cuda --timeout-s 180`, via `modal run cloud/modal_runtime.py --tests`, run `20261106T185236Z`. All five remote stages exited 0 with no timeout.

GPU runtime: `{"runtime/compute_sanitizer": "/usr/local/cuda/bin/compute-sanitizer", "runtime/cuda_bindings_version": "13.5.0", "runtime/cuda_tile_version": "1.8.0", "runtime/cutlass_dsl_version": "4.8.1", "runtime/driver_ok": true, "runtime/driver_version": "610.57.04", "runtime/gpu0_compute_cap": "9.0", "runtime/gpu0_cutile_supported": true, "runtime/gpu0_name": "NVIDIA H200", "runtime/gpu0_uuid": "GPU-aac19e0f-0b54-352e-a3a3-015ad052e871", "runtime/gpu_count": 1, "runtime/ncu": "/usr/local/cuda/bin/ncu", "runtime/ninja": "/usr/local/bin/ninja", "runtime/nsys": "/usr/local/bin/nsys", "runtime/nvcc_ok": true, "runtime/nvcc_version": "13.3", "runtime/ok_cuda_backend": true, "runtime/ok_cutedsl": true, "runtime/ok_cutile": true, "runtime/ok_tilelang": true, "runtime/ok_triton_backend": true, "runtime/python": "3.12.1", "runtime/python_ok": true, "runtime/tileiras_ok": true, "runtime/tileiras_reason": null, "runtime/tileiras_version": "13.3.36", "runtime/tilelang_version": "0.1.15", "runtime/torch_cuda_build": true, "runtime/torch_version": "2.14.1+cu130", "runtime/triton_version": "3.8.0", "runtime/unavailable_reason": null}`

[Results](artifacts/results-20261106T185236Z.json), [kernel names](artifacts/kernel_names-20261106T185236Z.json), [environment](artifacts/environment-20261106T185236Z.json), [unittest log](artifacts/gpu_tests.stderr-20261106T185236Z.log).

```text
Ran 30 tests in 380.509s
OK
```

| Fixture | Compile | Verify | Provenance | Reward | Reason |
|---|---|---|---|---|---|
| triton | passed | True | 1.0 | 2.0 | faster_than_eager |
| hack | passed | True | 0.0 | -1.0 | provenance |
| timeout | not completed | False | None | -1.0 | timeout |
| max_turns | not completed | False | None | -1.0 | max_turns |
| overlong | passed | True | 1.0 | 1.25 | faster_than_eager |
| cuda | passed | True | 1.0 | 2.0 | faster_than_eager |
| tilelang | passed | True | 1.0 | 2.0 | faster_than_eager |
| hack-tilelang | passed | True | 0.0 | -1.0 | provenance |
| cutile | passed | True | 1.0 | 2.0 | faster_than_eager |
| hack-cutile | passed | True | 0.0 | -1.0 | provenance |
| cutedsl | passed | True | 1.0 | 2.0 | faster_than_eager |
| hack-cutedsl | passed | True | 0.0 | -1.0 | provenance |

Raw profiler names from `python -m scripts.record_kernel_names --device cuda --timeout-s 600`, run `20261106T185236Z`. Every listed event is `agent`; every unmatched-name list is `[]`.

```text
cuda: void axpby_kernel<256>(float*, float const*, float const*, float, int)
triton: axpby_kernel
tilelang: axpby_tilelang_kernel
cutile: axpby_cutile_kernel_Kt1_A1f32_1i16t1_p16_A1f32_1i16t1_p16_A1f32_1i16t1_p16_Sf32_I128
cutedsl: kernel_cutlass_axpby_cute_kernel_tensorptrf32gmemo151201i64512_tensorptrf32gmemo151201i64512_tensorptrf32gmemo151201i64512_tensor00o1512011105121____0
```

No backend was unavailable. All five honest rollouts verified with provenance 1.0; all four Torch-op hacks verified with provenance 0.0 and reward -1. Timeout and max-turn fixtures returned their expected reasons.

GPU benchmark and graph tests passed. Raw benchmark latency series, Nsight/sanitizer runs, Chrome traces, other GPUs and broader workloads are not established by this run. Generic-name spoofing remains open pending classifier approval. The environment packaging fix was made after this remote upload and still needs a remote packaging check.
