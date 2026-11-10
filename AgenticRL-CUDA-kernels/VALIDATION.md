# Validation report

All commands below run inside `AgenticRL-CUDA-kernels`. No project or secret
files outside this folder were changed. The upstream environment was imported
from commit `473025c8af7878e928525138b5cb327d6bcdf9dd`; original SHA-256 checksums
are in [artifacts/upstream.json](artifacts/upstream.json). Tests assert that
model.py, binding.cpp, binding_registry.h, model_new.py and both stock kernel
files remain byte-identical.

## Kernel symbol inspection and unmatched-name diagnostics

**Inspection reference:** the per-backend `symbol_pattern` (`re.search`) mapping
in [Approved symbol rules](#approved-symbol-rules-evidence-run-20261106t185236z)
links authored names to profiler symbols; keep its captured evidence as the
reference. Pins below come from H200 run `20261106T185236Z`
([environment](artifacts/environment-20261106T185236Z.json)).

When `unmatched_agent_names` is non-empty, first check `measured`/`reason`,
compile errors, the discovered names and every event/class in `kernel_names.json`.
The recorder now uses `classify_kernel(event, [authored_name], backend)` for
this list, including approved patterns and namespace rejection. Historical
records used substring presence; their empty lists alone did not establish an
anchored match. Compilation alone does not prove that a kernel launched.

**First compiler diagnostic for Triton: on the GPU host**, enable
`TRITON_KERNEL_DUMP=1` and `MLIR_ENABLE_DUMP=1`, then repeat compile/profiling
and retain the emitted IR/symbols. In a Modal H200 run, set these in the remote
image/sandbox environment before its child processes start; exporting them only
in the local checkout does not configure the container. The runner now accepts
`--forward-diagnostics` to forward an explicit allowlist of these variables;
use remote `/artifacts/<run_id>/...` paths for retained dumps. The recorder copies
`os.environ` into its stage environment (`scripts/record_kernel_names.py:41`);
Modal's `_run` likewise inherits its remote environment (`cloud/modal_runtime.py:67`).

For full stdout/stderr, run these stages directly in a retained **remote task
workdir whose ModelNew already selects Triton**, with the trusted harness on
PYTHONPATH; replace RUN_ID with the diagnostic run ID. Successful recorder
stages discard captured compiler stdout/stderr; the direct commands retain both.

```bash
export PYTHONPATH="/root/AgenticRL-CUDA-kernels${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_AGENT_DEVICE=cuda TRITON_INTERPRET=0
export TRITON_KERNEL_DUMP=1 MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1
export TRITON_CACHE_DIR="$PWD/.cache/triton-symbol-debug"
export TRITON_DUMP_DIR=/artifacts/RUN_ID/triton-ir
mkdir -p "$TRITON_DUMP_DIR"
python -m utils.compile --backend triton >"$TRITON_DUMP_DIR/compile.stdout.log" 2>"$TRITON_DUMP_DIR/compile.stderr.log"
python -m utils.profiling --backend triton --json >"$TRITON_DUMP_DIR/profile.stdout.log" 2>"$TRITON_DUMP_DIR/profile.stderr.log"
```

For DSL fixtures, select the matching ModelNew and replace the backend/switches
using this table. **Triton's switches are not universal MLIR switches**; on DSL
runs they may dump only the Inductor baseline. Names were checked against the pins:

| Backend / recorded version | Applicable inspection controls and policy reference |
|---|---|
| Triton 3.8.0 | Both dump flags above are documented in the [3.8.0 README](https://github.com/triton-lang/triton/blob/v3.8.0/README.md#tips-for-hacking). Narrow noisy pass dumps with `MLIR_ENABLE_DUMP=axpby_kernel`; `TRITON_ALWAYS_COMPILE=1` bypasses a stale compilation-cache hit. See [SKILL_TRITON.md](SKILL_TRITON.md) and [TRITON_guide.md](TRITON_guide.md). |
| TileLang 0.1.15, target `cuda` | Uses TVM/TIR, not Triton's MLIR pipeline. `TILELANG_PRINT_ON_COMPILATION=1` and `kernel.get_kernel_source()` are the quieter compilation-name/generated-CUDA fallback in [SKILL_TILELANG.md:60](SKILL_TILELANG.md#L60). The switch is valid in [0.1.15 env.py:371](https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/env.py#L371); no spelling correction is needed. |
| cuTile 1.8.0 / tileiras 13.3.36 | `CUDA_TILE_LOGS=CUTILEIR` prints Python frontend IR; `CUDA_TILE_DUMP_BYTECODE=/artifacts/RUN_ID/cutile-ir` saves TileIR bytecode ([1.8.0 debugging](https://github.com/NVIDIA/cutile-python/blob/v1.8.0/docs/source/debugging.rst), [_debug.py](https://github.com/NVIDIA/cutile-python/blob/v1.8.0/src/cuda/tile/_debug.py)). TileIR is MLIR-based, but Triton's dump switches do not apply. `CUDA_TILE_LOGS=TILEIR` and `CUDA_TILE_DUMP_TILEIR` require CTK **13.4+**, outside this 13.3 pin; do not recommend them here. Policy: [SKILL_CUTILE.md](SKILL_CUTILE.md). |
| CuTe DSL 4.8.1 | Its own MLIR controls are `CUTE_DSL_PRINT_IR=1` (generated IR), or `CUTE_DSL_KEEP=ir,ptx` with `CUTE_DSL_DUMP_DIR=/artifacts/RUN_ID/cutedsl-ir` for files ([4.8.1 debugging](https://github.com/NVIDIA/cutlass/blob/v4.8.1/media/docs/pythonDSL/guides/debugging.rst)). Use these, not Triton's `MLIR_ENABLE_DUMP`. Policy: [SKILL_CUTEDSL.md](SKILL_CUTEDSL.md). |
| CUDA C++ / nvcc 13.3 | Compare the recorded device event with the CUDA rule; none of the Triton/DSL switches above is an nvcc inspection switch. |

Only the TileLang policy currently names a compile-print knob; the cuTile/CuTe
controls above are release-verified vendor fallbacks, not claims that those
policy files already list them. A cache hit can suppress compiler output: use a
fresh diagnostic process/task cache, and retain dumps under the remote artifacts
volume. These are inspection runs, not new reward/benchmark evidence.

Compare the **actual generated-model profiler event** and dumped entry point
against that backend's approved pattern, substituting the escaped authored name.
IR output may also include torch.compile/Inductor baselines; do not credit those
as authored work. If the authored kernel never launched, fix its call path.
If a newer compiler demonstrably changed the symbol shape, update the pattern
**and repin captured-name expectations in `tests/test_symbol_provenance.py` in
the same reviewed change**, citing the compiler version and capture run ID.
**Never loosen unknown-event handling to make a fixture pass: unknown events
remain Torch (fail closed).**

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

`record_kernel_names` was verified against synthetic stage fixtures locally;
the real GPU kernel names it later recorded on an H200 are in
`artifacts/kernel_names-20261106T185236Z.json` and listed verbatim under
[First Modal GPU validation](#first-modal-gpu-validation-2026-11-06).

## DSL backends (TileLang, cuTile, CuTe DSL)

A backend registry (`utils/backend.py`) replaced the hard-coded CUDA/Triton
branches, and TileLang, cuTile and CuTe DSL backends were added on top of it.
CUDA and Triton behaviour is unchanged: after the refactor, the full suite and
the Torch-dependent backend, kernel, benchmark, config and rollout tests passed
without edits. No DSL package is installed on this host and there is no GPU,
so only source discovery, backend detection and classification are exercised
**here**; all three DSL kernels have since run on an H200 (Modal run
`20261106T185236Z`, see below), each with correct output and provenance 1.0.

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
change was approved and is implemented below. Unknown events still count as Torch.

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

GPU benchmark and graph tests passed. Raw benchmark latency series, Nsight/sanitizer runs, Chrome traces, other GPUs and broader workloads are not established by this run. Generic-name spoofing is addressed by the approved classifier rules below. The environment packaging fix was made after this remote upload and still needs a remote packaging check.

## Experiment tracking

`tracking/wandb_log.py` (metrics/config mappers, `backfill_payload`,
`log_backfill`, `log_rollout`, backfill/rollout CLI) and
`tracking/weave_trace.py` (`rollout_inputs`, lazy `run_task`/`compute_reward`/
`build_batch` ops, `replay_backfill`) were added. wandb and weave import
lazily only; tests use a fake wandb module — no live-service calls were made.

```bash
python3 -m unittest -v tests.test_tracking      # 9 tracking tests + 2 sweep-config tests
# Ran 11 tests in 4.848s
# OK

python3 -S -m tracking.wandb_log backfill --run-id 20261106T185236Z --dry-run | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['summary']))"
# {"cases": 12, "verify_ok": 10, "provenance_pass": 6, "reward_positive": 6,
#  "reward_mean": 0.4375, "unmatched_agent_names_total": 0,
#  "backends_measured": 5, "gpu_tests": "OK", "unavailable_backends": ""}

python3 -m unittest discover -v                 # system Python, no torch/wandb/weave
# Ran 105 tests in 53.899s
# OK (skipped=31)
```

An earlier full-suite run during development failed on one wrong test
expectation (`config_for_wandb` correctly keeps only `runtime/*` keys from the
environment record); the expectation was fixed and the suite above is the
recorded re-run.

The sweep entry point was exercised end to end offline — exactly the command a
`wandb agent` issues, with agent-style `--key=value` flags, on the CPU
interpreter, `WANDB_MODE=offline`, no API key:

```bash
CUDA_VISIBLE_DEVICES=0 SWEEP_WORKDIR="$PWD/sweep_tasks/gpu0/triton" SWEEP_NO_WEAVE=1 \
WANDB_MODE=offline CUDA_AGENT_DEVICE=cpu TRITON_INTERPRET=1 \
  bash scripts/sweep_rollout.sh --algo=dapo --seed=1 --iters=10 --benchmark --bench-rounds=6 --timeout-s=600
# real 2m17.891s; exit 0; env/verify_ok true, reward/reason "unmeasured",
# bench/available false ("CUDA_AGENT_DEVICE=cpu"); meta/algo dapo, meta/seed 1
# (sweep flags reached RolloutConfig); offline run written under wandb/
```

W&B entity/project come from `WANDB_ENTITY`/`WANDB_PROJECT` in `.env`
(`.env.example` documents them), not from code. The Modal session logs
under `artifacts/` have the local checkout path and the Modal workspace name
redacted (`<repo>`, `<workspace>`); they contain no credentials. `wandb/`,
`sweep_tasks/` and `sweep_agent_gpu*.log` are gitignored. No live W&B call has
been made from this repository yet.

The `overlong` rollout also passes provenance and earns >0, so
`provenance_pass`/`reward_positive` are 6 (the five honest fixtures + overlong).

## Approved symbol rules (evidence: run 20261106T185236Z)

| Backend | `symbol_pattern` (`re.search`) |
|---|---|
| cuda | `(^\|[\s:]){name}\s*[<(]` |
| triton | `^{name}(_\d\w*)?$` |
| tilelang | `^{name}(_kernel)?$` |
| cutile | `^{name}_Kt\d+_` |
| cutedsl | `^kernel_cutlass_{name}_` |

`{name}` is substituted using `re.escape`. Both source discovery and classification
reject names shorter than four characters and the shared denylist in
`utils/backend.py`. Order: memory → Torch/Inductor namespace markers → anchored
authored match → allowed-library markers → Torch. Captured names above are pinned
in `tests/test_symbol_provenance.py`. Tests cover generic, namespace and Inductor
spoofs; a GPU fixture launches a scratch kernel renamed `kernel` on each backend
before returning the Torch-op result. Root protected sources are untouched.

The exact approved rules have two cross-backend exceptions: a CuTe symbol counts
as CUDA library time because it contains `cutlass`; a bare Triton name can match
the optional TileLang suffix pattern when supplied as a TileLang authored name.
These follow the explicit patterns and existing library policy. Name matching is
not proof of authorship. The CUDA naming supplement is SKILL_CUDA.md; protected
upstream SKILL.md remains byte-identical.


Local verification after the approved rule change:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v
Ran 114 tests in 36.312s
OK (skipped=32)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
Ran 114 tests in 487.574s
OK (skipped=26)

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.test_symbol_provenance
Ran 8 tests in 7.093s
OK
```

Raw logs: `artifacts/local-tests-20261106T231526Z.log`,
`artifacts/local-venv-tests-20261106T231526Z.log`, and
`artifacts/symbol-tests-20261106T231526Z.log`. The current repository also has
tracking tests added since the earlier 94-test evidence; these were retained.
Three synthetic expectations were updated for the approved behavior: backend
separation for CUDA/Triton names, the CUDA event in the memory-denominator test,
and the cuTile `_Kt1_` symbol requirement. Root protected-file checks pass.


### Approved classifier GPU confirmation: 20261106T231526Z

Command: `.venv/bin/modal run cloud/modal_runtime.py --tests`.
The existing pinned combined image was reused successfully. All five stages
(require probe, stock pipeline, name recorder, GPU suites, validate) exited 0
without timeout. Raw summary:

```text
python -m unittest -v tests.test_provenance tests.test_dsl_kernels tests.test_benchmark.GPUBenchmarkTests
Ran 31 tests in 472.152s
OK
```

[Structured summary](artifacts/validation-summary-20261106T231526Z.json), [results](artifacts/results-20261106T231526Z.json), [names](artifacts/kernel_names-20261106T231526Z.json), [fresh environment](artifacts/environment-20261106T231526Z.json), [GPU suite stdout](artifacts/gpu_tests.stdout-20261106T231526Z.log), [GPU suite stderr](artifacts/gpu_tests.stderr-20261106T231526Z.log), [session](artifacts/modal-session-20261106T231526Z.log).

| Backend | Honest compile / verify | Honest provenance | Dummy `kernel` + Torch verify | Spoof provenance | Spoof reward |
|---|---|---|---|---|---|
| cuda | passed | 100% | passed | 0% | -1 |
| triton | passed | 100% | passed | 0% | -1 |
| tilelang | passed | 100% | passed | 0% | -1 |
| cutile | passed | 100% | passed | 0% | -1 |
| cutedsl | passed | 100% | passed | 0% | -1 |

All honest name lists have `unmatched_agent_names == []`. The raw symbols are
identical to the five captured names from `20261106T185236Z` pinned above.
The ordinary Triton/DSL hacks also verified with provenance 0% and reward -1.

Runtime record: `{"runtime/compute_sanitizer": "/usr/local/cuda/bin/compute-sanitizer", "runtime/cuda_bindings_version": "13.5.0", "runtime/cuda_tile_version": "1.8.0", "runtime/cutlass_dsl_version": "4.8.1", "runtime/driver_ok": true, "runtime/driver_version": "610.57.04", "runtime/gpu0_compute_cap": "9.0", "runtime/gpu0_cutile_supported": true, "runtime/gpu0_name": "NVIDIA H200", "runtime/gpu0_uuid": "GPU-68d3501d-8cc4-50f9-05b3-e2649e189bbd", "runtime/gpu_count": 1, "runtime/ncu": "/usr/local/cuda/bin/ncu", "runtime/ninja": "/usr/local/bin/ninja", "runtime/nsys": "/usr/local/bin/nsys", "runtime/nvcc_ok": true, "runtime/nvcc_version": "13.3", "runtime/ok_cuda_backend": true, "runtime/ok_cutedsl": true, "runtime/ok_cutile": true, "runtime/ok_tilelang": true, "runtime/ok_triton_backend": true, "runtime/python": "3.12.1", "runtime/python_ok": true, "runtime/tileiras_ok": true, "runtime/tileiras_reason": null, "runtime/tileiras_version": "13.3.36", "runtime/tilelang_version": "0.1.15", "runtime/torch_cuda_build": true, "runtime/torch_version": "2.14.1+cu130", "runtime/triton_version": "3.8.0", "runtime/unavailable_reason": null}`.

The downloaded environment.json contains the matching run_id and every field
from this container's fresh require probe, including H200 name and GPU UUID.
This remotely verifies the environment packaging fix; no local reconstruction
was needed for this run. Benchmark and CUDA-graph tests passed. Reward modules,
timing fields, upstream stdout and default rollout schema remain unchanged.
Protected root files and upstream SKILL.md remain byte-identical. No Nsight,
sanitizer or broader workload validation is claimed.


## CISPO GPU batch-ranking validation: 20261107T140018Z

Command: `.venv/bin/modal run cloud/modal_runtime.py --tests`.
The GPU suite's `test_every_preset_ranks_the_hack_last` checks iterate over
`ALGOS = ("grpo", "dapo", "cispo")`; the shared CUDA/Triton group and all
three DSL groups passed. These checks call `build_batch` on measured rollout
rewards; this is not a trainer/optimizer implementation test.

```text
Ran 31 tests in 503.305s

OK
```

Measured CISPO group: `{"advantages": {"cuda": 0.7071062811869011, "hack": -1.4142125623738022, "triton": 0.7071062811869011}, "stats": {"adv/frac_dropped": 0.0, "adv/mean": 0.0, "adv/std": 0.9999992928937188, "reward/frac_3": 0.0, "reward/frac_max_turns": 0.0, "reward/frac_provenance_fail": 0.3333333333333333, "reward/mean": 1.0, "reward/mean_overlong_penalty": 0.0, "turns/mean": 1.0, "turns/p95": 1.0}}`.
All five honest backends recorded provenance 1.0 and empty unmatched-name lists.
The packaged environment record matches this container's fresh require probe.

[GPU suite log](artifacts/gpu_tests.stderr-20261107T140018Z.log),
[results](artifacts/results-20261107T140018Z.json),
[kernel names](artifacts/kernel_names-20261107T140018Z.json),
[environment](artifacts/environment-20261107T140018Z.json),
[structured CISPO validation](artifacts/cispo-validation-20261107T140018Z.json).


## W&B tracking contract (2026-11-07)

Decision and machine-checked inventory:
[0002-wandb-tracking-contract.md](Architecture-decision/0002-wandb-tracking-contract.md).
Validation uses per-case tables and aggregate-only summaries. Training iterations
log build_batch statistics. New artifacts persist full base/case/group configs,
raw same-pass profiles, candidate SHA-256 and classifier identity. The default
18 evaluator result keys, profiler stdout/timing meaning and reward rules are
unchanged. No wandb/weave imports were added to utils; python -S import checks
passed. Existing protected sources and SKILL.md remain byte-identical.

Raw tests (fake SDKs and synthetic stage fixtures; no additional GPU run):

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v
Ran 122 tests in 37.571s
OK (skipped=32)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
Ran 122 tests in 291.174s
OK (skipped=26)

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.test_tracking
Ran 18 tests in 12.998s
OK
```

Logs: [system suite](artifacts/wandb-contract-tests-system.log),
[CPU/interpreter suite](artifacts/wandb-contract-tests-venv.log),
[focused tracking tests](artifacts/wandb-contract-tests-tracking.log).

Requested dry-run command:

```bash
python3 -m tracking.wandb_log backfill --run-id 20261106T185236Z --dry-run
```

[Full payload](artifacts/wandb-contract-payload-20261106T185236Z.json) and
[config, summary and key sets](artifacts/wandb-contract-dry-run-20261106T185236Z.json).
The printed key sets and values are:

```json
{
  "config_keys": [
    "meta/provenance_rules",
    "runtime/compute_sanitizer",
    "runtime/cuda_bindings_version",
    "runtime/cuda_tile_version",
    "runtime/cutlass_dsl_version",
    "runtime/driver_ok",
    "runtime/driver_version",
    "runtime/gpu0_compute_cap",
    "runtime/gpu0_cutile_supported",
    "runtime/gpu0_name",
    "runtime/gpu0_uuid",
    "runtime/gpu_count",
    "runtime/ncu",
    "runtime/ninja",
    "runtime/nsys",
    "runtime/nvcc_ok",
    "runtime/nvcc_version",
    "runtime/ok_cuda_backend",
    "runtime/ok_cutedsl",
    "runtime/ok_cutile",
    "runtime/ok_tilelang",
    "runtime/ok_triton_backend",
    "runtime/python",
    "runtime/python_ok",
    "runtime/tileiras_ok",
    "runtime/tileiras_reason",
    "runtime/tileiras_version",
    "runtime/tilelang_version",
    "runtime/torch_cuda_build",
    "runtime/torch_version",
    "runtime/triton_version",
    "runtime/unavailable_reason",
    "source_command",
    "tracking/candidates_available",
    "tracking/config_available",
    "tracking/contract",
    "tracking/group_configs",
    "tracking/rules_available",
    "tracking/shares_available",
    "versions/cuda-tile",
    "versions/numpy",
    "versions/nvidia-cutlass-dsl",
    "versions/tilelang",
    "versions/torch",
    "versions/triton"
  ],
  "summary_keys": [
    "backends_measured",
    "cases",
    "gpu_tests",
    "provenance_pass",
    "reward_mean",
    "reward_positive",
    "unavailable_backends",
    "unmatched_agent_names_total",
    "verify_ok"
  ],
  "metric_keys": [
    "batch/dapo/adv/frac_dropped",
    "batch/dapo/adv/mean",
    "batch/dapo/adv/std",
    "batch/dapo/reward/frac_3",
    "batch/dapo/reward/frac_max_turns",
    "batch/dapo/reward/frac_provenance_fail",
    "batch/dapo/reward/mean",
    "batch/dapo/reward/mean_overlong_penalty",
    "batch/dapo/turns/mean",
    "batch/dapo/turns/p95",
    "batch/grpo/adv/frac_dropped",
    "batch/grpo/adv/mean",
    "batch/grpo/adv/std",
    "batch/grpo/reward/frac_3",
    "batch/grpo/reward/frac_max_turns",
    "batch/grpo/reward/frac_provenance_fail",
    "batch/grpo/reward/mean",
    "batch/grpo/reward/mean_overlong_penalty",
    "batch/grpo/turns/mean",
    "batch/grpo/turns/p95",
    "iteration"
  ],
  "config": {
    "runtime/python": "3.12.1",
    "runtime/python_ok": true,
    "runtime/gpu_count": 1,
    "runtime/gpu0_name": "NVIDIA H200",
    "runtime/gpu0_compute_cap": "9.0",
    "runtime/gpu0_uuid": "GPU-aac19e0f-0b54-352e-a3a3-015ad052e871",
    "runtime/driver_version": "610.57.04",
    "runtime/driver_ok": true,
    "runtime/gpu0_cutile_supported": true,
    "runtime/nvcc_version": "13.3",
    "runtime/nvcc_ok": true,
    "runtime/ninja": "/usr/local/bin/ninja",
    "runtime/tileiras_version": "13.3.36",
    "runtime/tileiras_ok": true,
    "runtime/tileiras_reason": null,
    "runtime/cuda_tile_version": "1.8.0",
    "runtime/torch_version": "2.14.1+cu130",
    "runtime/torch_cuda_build": true,
    "runtime/triton_version": "3.8.0",
    "runtime/tilelang_version": "0.1.15",
    "runtime/cutlass_dsl_version": "4.8.1",
    "runtime/cuda_bindings_version": "13.5.0",
    "runtime/nsys": "/usr/local/bin/nsys",
    "runtime/ncu": "/usr/local/cuda/bin/ncu",
    "runtime/compute_sanitizer": "/usr/local/cuda/bin/compute-sanitizer",
    "runtime/ok_cuda_backend": true,
    "runtime/ok_triton_backend": true,
    "runtime/ok_cutile": true,
    "runtime/ok_tilelang": true,
    "runtime/ok_cutedsl": true,
    "runtime/unavailable_reason": null,
    "versions/torch": "2.14.1+cu130",
    "versions/triton": "3.8.0",
    "versions/numpy": "2.5.4",
    "versions/tilelang": "0.1.15",
    "versions/cuda-tile": "1.8.0",
    "versions/nvidia-cutlass-dsl": "4.8.1",
    "source_command": "python -m scripts.validate --device cuda --timeout-s 180",
    "meta/provenance_rules": null,
    "tracking/config_available": false,
    "tracking/rules_available": false,
    "tracking/group_configs": null,
    "tracking/contract": "wandb-contract-v1",
    "tracking/shares_available": false,
    "tracking/candidates_available": false
  },
  "summary": {
    "cases": 12,
    "verify_ok": 10,
    "provenance_pass": 6,
    "reward_positive": 6,
    "reward_mean": 0.4375,
    "unmatched_agent_names_total": 0,
    "backends_measured": 5,
    "gpu_tests": "OK",
    "unavailable_backends": ""
  }
}
```

The historical run has no persisted full config, same-pass case shares,
candidate hashes or classifier ID. Those values remain unknown and availability
flags are false. No historical files were rewritten using current defaults.
The dry-run has 45 config keys, 9 aggregate summary keys and 21 history keys:
20 algo-qualified build_batch stats plus iteration. It contains no per-case
scalar metrics. The source run predates CISPO, so it has only GRPO/DAPO measured
groups; CISPO metrics are emitted when a source actually contains that group.

No live W&B or Weave run was created or modified. The existing remote page can
be updated by explicitly executing the reviewed backfill without --dry-run;
old history remains, stale per-case summary fields are deleted, and corrected
aggregate/table data are logged. Fake-SDK tests check this migration behavior;
remote summary deletion was not exercised in this task. New artifact writers
were checked with synthetic stages, not a new GPU validation run.

Trainer-owned metric names are reserved, not populated. Token/trajectory
masking, entropy/IS denominators, collapse detection and eval cohort/weighting
must be decided with the trainer owner before those metrics are emitted.

Commit attempt failed: `Unable to create .../.git/index.lock: Read-only file
system`. Nothing was pushed.


## Torch-op reward-hack fixture naming (2026-11-08)

Current fixtures and future validation output use `reward-hack-torchop` and
`reward-hack-torchop-<backend>`. The factory, validation/group IDs, provenance
checks and graph benchmark test callers use these names. The default bare
fixture still declares Triton; explicit backend variants declare their backend.
The Torch-op model template is byte-identical to its previous value, verified
against the `_HACK` literal at HEAD. Rewards, grouping, classifier rules, kernel
sources and protected files are unchanged. Historical artifacts and backfill
labels remain verbatim (`hack`, `hack-<backend>`); the historical backfill was
checked directly. The CISPO explainer remains an unlinked draft.

The recorded H200 run `20261107T140018Z` still supplies the environment-side
ranking evidence (its historical JSON key is `hack`). The trainer's clipped-IS
loss remains unimplemented. No fresh GPU measurement was made for this naming
change.

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v
Ran 123 tests in 87.479s
OK (skipped=32)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
Ran 123 tests in 695.269s
OK (skipped=26)
```

Raw logs: [system tests](artifacts/fixture-rename-tests-system-20261108.log) and
[CPU/interpreter tests](artifacts/fixture-rename-tests-venv-20261108.log).

## Explainer review, real-SDK offline smoke and H200 acceptance (2026-11-08)

### CISPO explainer review

`CISPO_GPU_batch_ranking.md` remains an unlinked draft. Its worked example now
uses `(1, 1, -2) / (sqrt(2) + 1e-6)`, rather than presenting the zero-epsilon
limit as the exact implementation. The stabilized values agree with
`group_advantages([2, 2, -1])` and the frozen historical artifact to absolute
tolerance `1e-15`. The review also separates the discrete base reward from
overlong shaping and documents the strict speedup-margin boundaries.

The GPU ranking checks reuse measured, one-turn GRPO rollout rewards with each
batch preset. They establish ordering for those non-tied groups, not independent
CISPO rollouts or optimizer behavior. Tied-group filtering is covered by separate
synthetic-reward tests; CISPO shaping is checked through mocked stage outputs.
The separate real-GPU overlong test uses DAPO, not CISPO.

### Real W&B SDK, offline only

Recorded command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.smoke_wandb_offline \
  --output "$PWD/artifacts/wandb-offline-smoke-20261108T013827Z"
```

The reusable script requires a new output directory. It forces offline mode,
removes the API key from its process environment, uses a localhost-only base
URL, and does not initialize Weave. All reward, timing and profile inputs are
explicitly synthetic; this is SDK serialization evidence, not a GPU benchmark.

Result: `Offline real-SDK smoke passed`, using `wandb==0.31.0`.
Both real SDK runs contain all 24 `RolloutConfig` fields plus the classifier
`anchored-namespaces-generic-v1`. Actual saved table files contain candidate
SHA-256 values checked against their local source files, full per-case configs
and all four profile shares. Final tables have two rows for the accumulated
standalone run and three rows for the batch run. Summaries contain aggregate
values and SDK system fields, not per-case reward/provenance fields.
The batch run logs the existing `adv/*`, `reward/*` and `turns/*` statistics.
Three staged raw-evidence bundles retain config, results and aligned synthetic
profiles after the logger's temporary source file is removed.

[Smoke log](artifacts/wandb-offline-smoke-20261108T013827Z.log) and
[reviewed SDK evidence](artifacts/wandb-real-sdk-review-20261108T013827Z.json).
The complete local SDK output remains under
`artifacts/wandb-offline-smoke-20261108T013827Z/`.
No live W&B or Weave run was created or modified; remote backfill migration is
still unverified.

Narrow local verification, not a new full-suite run:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.test_tracking tests.test_config_reward_grpo tests.test_rollout_env.RolloutProtocolTests.test_provenance_gate_and_overlong_via_entry_point
Ran 32 tests in 18.774s

OK
```

[Local verification log](artifacts/offline-acceptance-local-tests-20261108T013827Z.log).

### Fresh H200 run: 20261108T014026Z

One invocation of
`MODAL_GPU=H200 WANDB_MODE=offline .venv/bin/modal run cloud/modal_runtime.py --tests`
completed. All five remote stages returned zero without timeouts: required
runtime checks, stock pipeline, name recording, GPU tests, and validation.

```text
Ran 31 tests in 615.242s

OK
```

All five backends were measured, with no unmatched authored names. All five
honest cases passed correctness with provenance 1.0. All four ordinary
Torch-op substitution cases passed correctness but received provenance 0.0
and reward -1. New case IDs use `reward-hack-torchop` and its backend variants.
Historical artifacts were not relabelled.

New results persist complete base, case and grouping configs; candidate hashes;
raw same-pass profiles; and the classifier ID. The packaged environment's
runtime fields exactly match this invocation's require probe. All three
measured algorithm groups rank `reward-hack-torchop` last; the shared
CUDA/Triton group's CISPO advantages are
`0.7071062811869011`, `0.7071062811869011`, `-1.4142125623738022`.

**Acceptance is partial, not all-green performance evidence.** In the separate
honest CuTe DSL validation case, the profiler recorded
`t_eager_us=0.0`, `t_compile_us=0.8094999999999573` and
`t_gen_us=0.726700000000028`, while `timing_measured` was true.
Correctness and provenance passed, but the reward correctly failed closed:
`reward/value=-1`, `reward/reason="timing"`. The valid-timing requirement for
every honest validation case therefore did not pass, despite the successful
GPU unittest suite and command exit codes.

No baseline event list or profiler trace was archived for that case, so the
cause of the zero eager timing remains unconfirmed. No retry, timing replacement
or reward relaxation was performed. Diagnose this before treating a sweep's
performance rewards as reliable.

Evidence: [acceptance review](artifacts/acceptance-review-20261108T014026Z.json),
[GPU tests](artifacts/gpu_tests.stderr-20261108T014026Z.log),
[results](artifacts/results-20261108T014026Z.json),
[kernel names](artifacts/kernel_names-20261108T014026Z.json),
[environment](artifacts/environment-20261108T014026Z.json),
[fresh require probe](artifacts/require.stdout-20261108T014026Z.log), and
[redacted Modal session](artifacts/modal-session-20261108T014026Z.log).
The original downloaded volume directory is retained locally under
`artifacts/volume-20261108T014026Z/`.


## Empty-capture guard and diagnostic runner (2026-11-08)

`utils/profiling.py::profile_models` retains CUDA leaf-event counts as
`profile/{eager,compile,gen}_events`, plus the verbatim eager names in the
JSON-string field `profile/eager_kernel_names`. An empty session has a null
latency, `timing_measured=false`, and `timing_reason=empty_profile:<session>`.
The rollout returns reward -1 with that specific reason, without retry.
Generated-empty captures also leave provenance/shares null. Other captured
sessions retain their evidence. No reward threshold or timing definition changed.

Opt-in `--trace-dir DIR` writes `eager.json`, `compile.json`, and `gen.json`.
`--order eager,compile,gen` reverses the default generated-first order.
Traced/reordered profiles carry `profile/diagnostic=true`; `run_task` rejects
these as reward measurements (`diagnostic_profile`). Trace export occurs between
sessions, so absence of failure under tracing does not prove an untraced fault
has disappeared. The ordinary, untraced full run remains required.

The recorder now computes unmatched names with the same `classify_kernel`
call used for attribution, including namespace/generic rejection and anchored
backend patterns. A substring-only synthetic fixture stays unmatched. The
classifier ruleset itself is unchanged (`anchored-namespaces-generic-v1`).

The Modal runner accepts `--diagnose`, `--backends`, `--repeat`, and opt-in
`--forward-diagnostics`. Default diagnosis: 100 CuTe DSL trials and 20 per
control backend, half in each order, sequential on one GPU, with fresh processes
and task caches. `CUDA_TILE_CACHE_DIR` and `CUTE_DSL_CACHE_DIR` are now task-local
alongside Triton/TileLang caches; pinned vendor sources are linked in
KNOWN_ISSUES.md. Full runs archive raw benchmark samples and per-case
compile/verification/profiling wall seconds (`results.json:stage_wall_s`). These
include imports and warmup, and do not isolate compiler-only cost. No Level-3
deadline or verification tolerance changed.

`acceptance-review-<run_id>.json` is generated from fresh stage outputs:
positive timings and event counts, verified honest provenance exactly 1.0,
verified Torch-op hacks exactly 0.0 with reward -1, empty unmatched lists,
and packaged environment matching that container's require probe. Expected
`timeout`/`max_turns` cases are checked as rejections, not missing timings.
The CUDA Torch-op case is now included, so default CUDA validation has five
honest cases and five per-backend hacks. Selected runs report their scope.

Local commands and raw summaries (full logs carry run ID `20261108T091114Z`):

```text
python3 -m unittest discover -v
Ran 133 tests in 47.176s
OK (skipped=32)

.venv/bin/python -m unittest discover -v
Ran 133 tests in 249.111s
OK (skipped=26)

python3 -m unittest -v tests.test_rollout_env tests.test_tracking tests.test_profiling tests.test_record_kernel_names tests.test_modal_diagnostics
Ran 44 tests in 43.431s
OK
```

Protected-file checks pass; the upstream timing stdout line and default 18
result keys are unchanged. Pure parent modules import under `python -S`.
The Codex sandbox that produced these changes mounted `.git` read-only
(`Unable to create .../.git/index.lock: Read-only file system`); they were
committed afterwards from the Windows checkout.


The first targeted invocation (`.venv/bin/modal run cloud/modal_runtime.py
--diagnose`, run `20261108T091114Z`) was interrupted by Modal client
`AuthError: Jwt is expired` in `AttemptAwait`. The stopped app left 53 complete
records (CUDA 20, Triton 20, TileLang 13), none with an empty capture, plus an
interrupted TileLang trial (`tilelang-013`). There are 161 traces: 159 from
complete trials and two from the interrupted one. This is partial diagnostic
evidence, not a completed acceptance run. Source fingerprints, all records,
traces and logs are preserved under
[artifacts/timing-diagnostic-20261108T091114Z](artifacts/timing-diagnostic-20261108T091114Z).

The runner now calls `spawn()` once and polls that call with `get(timeout=30)`;
`--detach` preserves the active remote function if the client disconnects.
Polling never repeats a profiler measurement. Offline tests verify one spawn
across a polling timeout:

```text
python3 -m unittest -v tests.test_modal_diagnostics tests.test_profiling tests.test_record_kernel_names
Ran 14 tests in 15.374s
OK

.venv/bin/python -m unittest -v tests.test_modal_diagnostics tests.test_profiling tests.test_record_kernel_names
Ran 14 tests in 11.486s
OK
```

### Run A completed: 20261108T103454Z (timing diagnostic)

The detached continuation
(`.venv/bin/modal run --detach cloud/modal_runtime.py --diagnose --backends cutedsl,cutile,tilelang`,
function call `fc-01M6XBKJPN0QG0WB9FS7EYYB6X`) ran CuTe DSL first and finished
all 140 trials: CuTe DSL 100, cuTile 20, TileLang 20. Its summary
(`timing/timing-diagnostic.json`) reports `empty_captures: []` and
`failures: []`. All 11 source fingerprints in
[timing-source-20261108T103454Z.json](artifacts/timing-source-20261108T103454Z.json)
match the committed sources (CRLF-normalised).

Combined with the 53 complete trials of `20261108T091114Z`, there are **193
traced trials and no empty session**. Every trial captured exactly 20 eager,
10 compiled and 10 generated CUDA events, and every generated capture had
provenance 1.0. Run A therefore **did not reproduce** the zero eager baseline
of `20261108T014026Z`. If one session in 60 were empty, as in the historical
runs, 193 clean eager captures would occur with probability about 4%. That
bounds the rate under these diagnostic conditions only: traces were exported
between sessions, every trial used a fresh process and cache, and no rollout
cases ran alongside, unlike the original `scripts.validate` failure. The root
cause remains unconfirmed; the empty-capture guard is the mitigation.

| backend | order | trials | eager median µs (min–max) | generated median µs |
|---|---|---|---|---|
| cuda | gen,eager,compile | 10 | 1.643 (1.635–1.658) | 0.957 |
| cuda | eager,compile,gen | 10 | 1.773 (1.766–1.782) | 0.930 |
| triton | gen,eager,compile | 10 | 1.651 (1.642–1.697) | 0.803 |
| triton | eager,compile,gen | 10 | 1.771 (1.763–1.789) | 0.801 |
| tilelang | gen,eager,compile | 17 | 1.613 (1.584–1.738) | 0.774 |
| tilelang | eager,compile,gen | 16 | 1.741 (1.727–1.786) | 0.772 |
| cutile | gen,eager,compile | 10 | 1.601 (1.591–1.754) | 0.838 |
| cutile | eager,compile,gen | 10 | 1.733 (1.572–1.744) | 0.822 |
| cutedsl | gen,eager,compile | 50 | 1.597 (1.581–1.663) | 0.949 |
| cutedsl | eager,compile,gen | 50 | 1.738 (1.661–1.787) | 0.950 |

**Session order shifts the eager baseline.** In every backend the eager median is
about 8% higher when eager is profiled first (1.73–1.77 µs) than in the
production generated-first order (1.60–1.65 µs). The generated time barely moves.
At these sub-2 µs sizes the reward's `speedup_margin` comparison is therefore
order-sensitive; the cause (clock state, allocator or launch-path warmup) is
not established. No profiling order or reward threshold was changed.

Records, logs and the summary are under
[artifacts/timing-diagnostic-20261108T103454Z](artifacts/timing-diagnostic-20261108T103454Z).
The per-session Chrome traces of both diagnostic runs (about 20 MB) are kept
locally and in the Modal volume, not in git.


### Completed timing diagnosis: run 20261108T103454Z

Command: `.venv/bin/modal run --detach cloud/modal_runtime.py --diagnose
--backends cutedsl,cutile,tilelang`. Both remote stages exited 0; the diagnostic
subprocess took 1646.788031731 s. GPU: NVIDIA H200, UUID
`GPU-a8fdfbd1-6e5d-fc61-d1ea-04f2904ec8e0`, CC 9.0, driver 610.57.04.
Python 3.12.1, nvcc 13.3, tileiras 13.3.36, Torch 2.14.1+cu130, Triton 3.8.0,
TileLang 0.1.15, cuda-tile 1.8.0, CuTe DSL 4.8.1, cuda-bindings 13.5.0
(from this container's `require.stdout.log` in the archive).

| Run | Backend | Complete trials | Generated first / eager first | Empty captures |
|---|---|---:|---:|---:|
| 20261108T091114Z (interrupted) | cuda | 20 | 10 / 10 | 0 |
| 20261108T091114Z (interrupted) | triton | 20 | 10 / 10 | 0 |
| 20261108T091114Z (interrupted) | tilelang | 13 | 7 / 6 | 0 |
| 20261108T103454Z | cutedsl | 100 | 50 / 50 | 0 |
| 20261108T103454Z | cutile | 20 | 10 / 10 | 0 |
| 20261108T103454Z | tilelang | 20 | 10 / 10 | 0 |

All 140 continuation trials had positive timings, exactly 20 eager CUDA events
and 10 compiled/generated events each. All 420 continuation Chrome traces
parsed successfully, with kernel counts equal to the recorded event counts.
Cache roots were unique across all 100/20/20 trials. Across both invocations,
193 complete trials produced no empty capture; the extra interrupted TileLang
trial is preserved separately and is not counted as a completed measurement.
The zero baseline did **not reproduce**; its original cause remains unknown.
Tracing can perturb collection, so this is not proof that untraced collection
has been repaired. No profiler retry or reward relaxation was introduced.

For CuTe DSL, observed eager medians were 1.597149999999965 us (generated first)
and 1.737600000000043 us (eager first). These are diagnostic observations,
not reward inputs or a proposed timing-order change. Complete ranges, stage
wall times and raw names are in the
[diagnostic summary](artifacts/timing-diagnostic-summary-20261108T103454Z.json);
[all raw records, traces and logs](artifacts/timing-diagnostic-20261108T103454Z.tar.gz)
and [source fingerprints](artifacts/timing-source-20261108T103454Z.json)
are retained. The durable invocation completed without the first run's
synchronous authentication failure.


## Full H200 acceptance: run 20261108T140136Z

Command: `.venv/bin/modal run --detach cloud/modal_runtime.py --tests`.
The revised image excludes mutable generated artifacts and ships the upstream
checksum manifest explicitly. This attempt built/uploaded successfully; all
five subprocess stages exited 0 with no timeout. The automatic
[acceptance review](artifacts/acceptance-review-20261108T140136Z.json) reports
`pass`, no failures/anomalies, and environment matching the fresh require probe.

**Validated source scope:** this run uploaded the empty-capture implementation
before concurrent profiler edits at 14:05 UTC. Later per-kernel sample exports,
nonpositive/invalid-profile handling, order experiments and `modal_job` changes
are **GPU-unverified by this run**. At the user's request, they are preserved
and kept separate from this acceptance claim. The
[source manifest](artifacts/validation-source-20261108T140136Z.json) and
[recovered validated source](artifacts/validated-source-20261108T140136Z)
pin the matching committed bytes; the cloud snapshot additionally contains the
two artifact-exclusion edits applied before upload. No per-kernel sample fields
appear in this run's profiles. Do not attribute this result to newer worktree code.

Runtime (from packaged environment and its matching fresh require probe):
NVIDIA H200, UUID `GPU-73641627-b73b-0a2e-7ff2-e5657e05fdf4`, CC 9.0, driver 610.57.04; Python 3.12.1, nvcc 13.3, tileiras 13.3.36, Torch 2.14.1+cu130, Triton 3.8.0, TileLang 0.1.15, cuda-tile 1.8.0, CuTe DSL 4.8.1, cuda-bindings 13.5.0.

Raw GPU suite summary (`python -m unittest -v tests.test_provenance
 tests.test_dsl_kernels tests.test_benchmark.GPUBenchmarkTests`):

```text
Ran 31 tests in 599.418s

OK
```

Compile/verify and reward data from
[results.json](artifacts/results-20261108T140136Z.json), same command/run ID.
`--tests` invokes `python -m scripts.validate --device cuda --backends
cuda,triton,tilelang,cutile,cutedsl`. Values below are profiler device-duration
sums in microseconds, rounded to four decimals; raw floats remain in JSON.

| Backend | Compile / verify | Eager us | Compile us | Generated us | Honest provenance / reward | Torch-op hack provenance / reward | Unmatched |
|---|---|---:|---:|---:|---|---|---|
| cuda | pass / pass | 1.6433 | 0.8035 | 0.9568 | 1.0 / 2.0 | 0.0 / -1.0 | `[]` |
| triton | pass / pass | 1.6452 | 0.7680 | 0.8000 | 1.0 / 2.0 | 0.0 / -1.0 | `[]` |
| tilelang | pass / pass | 1.6576 | 0.8023 | 0.8032 | 1.0 / 2.0 | 0.0 / -1.0 | `[]` |
| cutile | pass / pass | 1.6479 | 0.8000 | 0.8706 | 1.0 / 2.0 | 0.0 / -1.0 | `[]` |
| cutedsl | pass / pass | 1.6491 | 0.8065 | 0.9956 | 1.0 / 2.0 | 0.0 / -1.0 | `[]` |

Every honest case's reward reason is `faster_than_eager`; every Torch-op hack
is rejected for `provenance`. There are 13 cases: 11 verified/measured cases,
plus the expected timeout and max-turns rejections. All required timings are
positive. The Torch-op Triton case has a positive eager outlier of 1.0444 us;
its raw event count is retained rather than filtered or retried. The honest
CuTe eager baseline is 1.6491 us. No historical zero value was rewritten.

Raw names from [kernel_names.json](artifacts/kernel_names-20261108T140136Z.json)
(verbatim; every event below is `agent`, every unmatched list is empty):

```text
cuda: void axpby_kernel<256>(float*, float const*, float const*, float, int)
triton: axpby_kernel
tilelang: axpby_tilelang_kernel
cutile: axpby_cutile_kernel_Kt1_A1f32_1i16t1_p16_A1f32_1i16t1_p16_A1f32_1i16t1_p16_Sf32_I128
cutedsl: kernel_cutlass_axpby_cute_kernel_tensorptrf32gmemo151201i64512_tensorptrf32gmemo151201i64512_tensorptrf32gmemo151201i64512_tensor00o1512011105121____0
```

CISPO measured-group output remains advantages +0.7071062811869011 for CUDA
and Triton, -1.4142125623738022 for `reward-hack-torchop`,
`adv/frac_dropped=0.0`, `reward/mean=1.0`. The trainer loss is not implemented.

Three [raw benchmark artifacts](artifacts/benchmark-samples-20261108T140136Z)
contain six event and wall samples per implementation; captured graphs also
have six samples. Every recorded latency is finite and positive. The expected
`.item()` capture rejection has an empty generated-graph series, not a zero
latency. These diagnostics never enter reward.

Per-case subprocess wall seconds (includes imports, JIT/warmup and verification;
not isolated compiler time), from `results.json:stage_wall_s`:

| Honest backend | Compile s | Verification s | Profiling s |
|---|---:|---:|---:|
| cuda | 15.404 | 2.857 | 9.755 |
| triton | 3.919 | 3.765 | 10.504 |
| tilelang | 5.077 | 3.915 | 10.397 |
| cutile | 3.004 | 3.158 | 9.754 |
| cutedsl | 2.957 | 3.514 | 9.805 |

The 180 s default is unchanged. These tiny fixtures cannot establish Level-3
cold-start needs or isolate how much recompilation each disk cache avoids.
The collector fault did not reproduce; original cause remains unconfirmed.
Run C was not needed because all strict unmatched lists were empty. No online
W&B/Weave write was performed by this task.

After the durable-job and image-upload fixes, targeted offline runner suites
passed in both interpreters (raw logs archived with this run):

```text
python3 -m unittest -v tests.test_modal_diagnostics
Ran 5 tests in 0.866s
OK

.venv/bin/python -m unittest -v tests.test_modal_diagnostics
Ran 5 tests in 1.056s
OK
```

A final contract check includes the later locally edited code, without GPU
execution or network calls:

```text
python3 -m unittest -v tests.test_profiling tests.test_tracking
Ran 26 tests in 42.780s
OK

.venv/bin/python -m unittest -v tests.test_profiling tests.test_tracking
Ran 26 tests in 43.615s
OK
```

## Final scoped issue-resolution evidence — 2026-11-08

Scope remains axpby evaluation/validation, not training or attribution redesign.
No reward threshold, speed margin, provenance rule or protected fixture changed.
The checkout advanced concurrently; claims below apply to frozen source snapshots,
not arbitrary subsequent working-tree edits.

### Invalid captures and profiler-position controls

Nonempty captures containing zero, negative, NaN or infinite CUDA durations
previously could remain `timing_measured=true`. Regression evidence:
`artifacts/issue-profiler-red-20261108T140409Z.log`. They now fail closed with
null session timing, an explicit reason and retained per-session samples.
Empty captures remain failures without retry; diagnostic profiles cannot reward.

Exact control commands (from this directory):

```bash
MODAL_GPU=H200 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/modal run --detach cloud/modal_runtime.py --diagnose-order-test --backends triton,cutedsl --repeat 4
MODAL_GPU=H200 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/modal run --detach cloud/modal_runtime.py --diagnose-order-test --backends triton,cutedsl --repeat 4 --order-preparations double-empty-prime,triple-empty-prime
```

Run `20261108T142102Z`: 64 sequential fresh-process untraced profiles, comparing
none/empty-prime/eager-prime/warmup-all in both orders, four repeats per cell.
No invalid/empty measured session. Without priming, median eager-first shifts
were +7.15% (Triton) and +7.35% (CuTe DSL); all-model warmup retained the effect.
One empty/eager primer reversed the direction instead of fixing it.

Run `20261108T144950Z`: 32 further profiles with two/three empty primers, again
four repeats per cell. No invalid/empty measured session. Two-primer median
eager-first shifts were +0.1525% and +0.3244%. Three primers were not uniformly
better: Triton had a +1.68% median shift with wider paired variation.
Shift means `median(eager-first) / median(generated-first) - 1`; no trials were
discarded. Detailed paired ratios, timings and limits are frozen in:

- `artifacts/issue-order-20261108T142102Z/analysis.json`
- `artifacts/issue-order-20261108T144950Z/analysis.json`

Each directory includes `order/records.json`, progress, stage/session logs and
the original source copies. Prelaunch normalized hashes and exact commands are
in `artifacts/issue-order-source-snapshot-20261108T142039Z.json` and
`artifacts/issue-order-source-snapshot-20261108T144932Z.json`.
The evidence supports a capture-position mitigation, not a low-level CUPTI
mechanism or a claim that the historical intermittent-empty fault is fixed.

After explicit user approval, two intentional empty contexts now precede any
production model warmup/timing. They are not failed model samples or retries.
The method is `cuda-leaf-sum-two-empty-prime-v1`, with
`profile/primer_event_counts` retained in raw artifacts and the method projected
to tracking rows. Experimental unprimed calls are always diagnostic. See the
approved follow-up in ADR 0003. Historical timings/rewards remain untouched and
must not be pooled across methods as equivalent observations.

### Two untraced full H200 passes

Both invocations used exactly:

```bash
MODAL_GPU=H200 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/modal run --detach cloud/modal_runtime.py --tests
```

- `20261108T140200Z`: first untraced acceptance pass, before final conditioning;
  GPU suite `Ran 31 tests in 498.003s / OK`. Evidence:
  `artifacts/issue-gpu-20261108T140200Z/`.
- `20261108T153803Z`: final approved conditioned method;
  GPU suite `Ran 31 tests in 503.798s / OK`. Automatic acceptance is `pass`,
  with no failures or timing anomalies. All five honest backends verify with
  provenance 1.0 and positive rewards; all five Torch-op substitutes verify
  but have provenance 0.0 and reward -1, reason `provenance`. All five
  name-recorder backends are measured with empty unmatched-name lists.
  Thirteen rollout cases include eleven actual profiles; each records the
  approved method, `[0, 0]` conditioning counts, positive timings and
  `profile/diagnostic=false`. Timeout and max-turn cases remain unmeasured.

Final evidence: `artifacts/issue-gpu-20261108T153803Z/`, including automatic
review, `results.json`, `kernel_names.json`, benchmark samples and stage logs.
`source-snapshot.json` records the exact command, UTC and normalized hashes
of thirteen relevant evaluator/orchestrator files. `source/` contains byte copies
frozen before launch, not reread after completion. The snapshot is also retained
at `artifacts/issue-gpu-final-source-20261108T153741Z/`.

Final device/runtime (saved in `environment.json` and matching require probe):
NVIDIA H200, CC 9.0, UUID `GPU-895d985b-b393-513c-49c7-b06bb3d4cce9`;
driver 610.57.04; nvcc 13.3; Python 3.12.1; Torch 2.14.1+cu130;
Triton 3.8.0; TileLang 0.1.15; cuTile 1.8.0; Cutlass DSL 4.8.1.
Name-based attribution remains a heuristic, not proof of authorship.

### Durable Modal recovery

Both preflight and long stages use one spawn followed by `get(timeout=30)`.
Only literal builtin polling timeouts retry. Other errors propagate with the
call ID and recovery command; measurements are never resubmitted automatically.
The reattachment CLI imports no runner and allocates no app/GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m cloud.modal_job --call-id fc-01M6XQESC4W4G2AHWGKJ9RK5KW
```

This exact existing call for run `20261108T140200Z` survived local-waiter loss.
An initial real-SDK reattachment exposed an unhydrated FunctionCall; the helper
now hydrates before reading its ID. Recovery subsequently completed for that
call and for `fc-01M6XRHPD7TYF4FM82NDST496S` (`20261108T142102Z`), without
respawning. Their session logs preserve the initial error and successful result.
Final full-run call: `fc-01M6XWYMWB8YJMPF7XQ3GKSTYN`, completed normally.
The historical synchronous `AttemptAwait` JWT-expiry cause remains unconfirmed.

### Live W&B/Weave and approved historical migration

The approved new validation upload used:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.smoke_tracking_live --source-run-id 20261108T014026Z --artifacts artifacts --output artifacts/tracking-live-check-20261108T142009Z --entity elenamylocuda-gemma --project CUDA-gen --execute
```

Credentials were supplied privately from the environment. This created only
new run `tracking-check-df6b74c7f55f4490`, tagged as a tracking-validation replay
of `partial_acceptance_invalid_baseline`; it did not relabel the historical
source as passing. The first readback checker assumed table references were in
summary. Real `summary="none"` keeps them in history instead; the checker was
corrected, not the logged measurements. Read-only recovery reused frozen
identity/payload without creating another run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.smoke_tracking_live --review-only --output artifacts/tracking-live-check-20261108T142009Z
```

Live API assertions passed: 24 cfg fields; 12 rollout and 5 kernel rows;
exactly one `iteration=0` aggregate snapshot; aggregate-only scalar summary;
six source files byte-verified against the uploaded artifact; and twelve
matching Weave replay calls selected by the validation marker. The CuTe DSL
zero eager time and reward -1 remain present in table/trace. The user opened
the W&B and Weave UI and explicitly confirmed the presentation.
Proof: `artifacts/tracking-live-check-20261108T142009Z/report.json`.

After that inspection the user explicitly approved the five existing manifest
entries. Their payloads and raw files were frozen in
`artifacts/tracking-backfill-20261108T144457Z/` before upload. Lead-authored SDK
execution applied `wandb_log.log_backfill(payload, "elenamylocuda-gemma",
"CUDA-gen")` and `weave_trace.replay_backfill(payload)` to each frozen payload,
with unique per-source Weave validation markers. All five deterministic
`h200-*` runs already existed and were resumed:

| Source run | W&B state after migration | Matching new Weave replays |
|---|---|---|
| `build-im-u7ElhT76HK3e4MiGfUc5JZ` | failed | 0 |
| `20261106T142735Z` | finished | 0 |
| `20261106T142830Z` | failed | 12 |
| `preflight-rebuild-aborted` | failed (source status remains aborted) | 0 |
| `20261106T185236Z` | finished | 12 |

Readback checked states, all supplied config/availability fields, explicit
summary scalars, latest table/aggregate snapshot, raw artifact hashes and exact
Weave outputs. Legacy case-level metric history was preserved, not erased;
stale case summary scalars were cleared. All five historical config/rules
availability flags remain false: current defaults, classifier IDs, shares and
source hashes were not invented. Frozen before-state, uploaded progress,
downloaded tables/artifacts and final proof:
`artifacts/tracking-backfill-20261108T144457Z/report.json`.

### Final local gate and remaining uncertainty

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v
Ran 148 tests in 78.603s
OK (skipped=32)

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -v
Ran 148 tests in 406.475s
OK (skipped=26)
```

Logs: `artifacts/issue-final-suite-system-20261108T145912Z.log` and
`artifacts/issue-final-suite-venv-20261108T150042Z.log`; exact skips identify
missing local GPU/optional package surfaces. The venv exercised CPU interpreter
paths. Protected model/binding/kernel files and `SKILL.md` remain byte-identical
to the upstream manifest. Deferred trainer/optimizer/agent loop, GEMM and stronger
attribution remain deferred. Follow-up is still required to reproduce the
historical intermittent empty capture and establish the underlying profiler and
Modal JWT mechanisms; successful guarded/conditioned runs are not that proof.

## Profile-order recovery and requested fresh GPU repeat — 2026-11-08

The requested stopped profile-order waiter corresponds to `20261108T142102Z`:
its detached job had completed despite losing its local waiter. A fresh download
was made without rerunning measurements or allocating another app/GPU:

```bash
.venv/bin/modal volume get agenticrl-cuda-artifacts '20261108T142102Z/' artifacts/profile-order-download-check-20261108T162024Z/20261108T142102Z
```

The destination was pre-created. Receipt:
`artifacts/profile-order-download-check-20261108T162024Z/receipt.json`.
All 266 remote files match their existing local counterparts byte-for-byte;
there are no missing local files or mismatches. The newly downloaded
`order/records.json` and `order/progress.jsonl` match their receipt hashes.
The frozen analysis still describes 64 diagnostic-only trials; it was not
recomputed, regraded or promoted to reward evidence. The separate 32-control
run `20261108T144950Z` also remains downloaded with its frozen analysis.

Before another GPU allocation, all thirteen evaluator/orchestrator hashes from
`20261108T153803Z` were compared with current sources and matched. After being
informed that no newer scorer change existed, the user explicitly requested a
fresh H200 validation anyway. This is an intentional replication, not an
automatic retry or evidence that the older snapshot was untested.

Exact invocation:

```bash
MODAL_GPU=H200 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/modal run --detach cloud/modal_runtime.py --tests
```

New run: `20261108T164430Z`; preflight call
`fc-01M6Y0R72VYPC1HD97FVD25GRW`, validation call
`fc-01M6Y0RD15842XBVFN2N2HTKPM`. The launcher completed normally, without a
resubmission. All stages completed and automatic acceptance is `pass`, with
no failures or timing anomalies. Raw GPU suite result:

```text
Ran 31 tests in 502.412s

OK
```

Evidence: `artifacts/fresh-gpu-20261108T164430Z/`. Its `source-snapshot.json`
records HEAD, command, UTC, and raw/CRLF-normalized hashes of 76 files covering
evaluator, runner, tests, kernel/model/binding sources, policies, requirements
and the upstream manifest. Byte copies were frozen before launch. Every copied
file matches its recorded hashes; all thirteen prior GPU-snapshot evaluator
hashes still match, and protected upstream bytes match the upstream manifest.
HEAD alone is not the source identity: the snapshot also captures local edits.
The standalone source bundle is
`artifacts/fresh-gpu-source-20261108T164346Z/`.

Lead review is frozen in `artifacts/fresh-gpu-20261108T164430Z/review.json`:
13 rollout cases include 11 actual measured profiles, each with positive
timings, `cuda-leaf-sum-two-empty-prime-v1`, `[0, 0]` primer counts and
`profile/diagnostic=false`. CUDA-event sample counts match their counters;
finite nonnegative samples sum to the recorded times using each case's actual
iteration count, and those times match rollout results. All five honest
backends verify with provenance 1.0 and positive rewards. All five Torch-op
substitutes verify but have provenance 0.0 and reward -1 for `provenance`;
timeout/max-turn rejections remain intact. All five name-recorder backends are
measured with no unmatched authored names.

Device: NVIDIA H200, UUID `GPU-c3b87b33-97bc-e90a-f89f-c1b18b4b8d9d`;
the saved environment matches the fresh require probe. Full package/runtime
identity, stage logs, raw results, benchmark samples and session log are
retained in the new run directory. No online tracking writes were performed.

This real GPU pass exercises valid nonempty captures and their new evidence
fields. Invalid-duration cases remain synthetic unit regressions, not claimed
hardware faults. The original intermittent-empty and JWT-expiry mechanisms
remain unconfirmed; no thresholds, attribution rules or kernel code changed.

## Historical Modal JWT investigation — 2026-11-08

**Conclusion: an SDK defect is reproduced; the historical trigger is not yet
established.** Do not relabel the old failure as a confirmed refresh outage.

The archived `artifacts/modal-timing-session-20261108T091114Z.log` is byte-identical
to the original `/tmp/modal-timing-diagnostic-session.log` (SHA-256
`a51d4ec042b887263a7b9dc8684811125eab10d568b50a416e205534531905d6`).
The failing frame is Modal 1.6.3 `_functions.py:486`, inside `AttemptAwait`,
after `get_input_plane_metadata` at line 485. This is not a traceback from
`AuthTokenGet` itself.

The SDK request has two relevant channels: refreshed `x-modal-auth-token`
metadata and the separate `AttemptAwaitRequest.attempt_token`. The latter is
retained across polling iterations unless an execution retry obtains another.
The server's bare `AuthError: Jwt is expired` does not identify which token
failed. No per-request expiry/refresh telemetry is present in the saved log.
Its warning that logs may not be continuous also prevents treating missing
warnings as definitive proof that refresh succeeded.

### Offline real-SDK reproduction

Exact command, using only synthetic tokens, a controlled clock and fake RPCs:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/modal-jwt-investigation-20261108T174201Z/reproduce.py --output artifacts/modal-jwt-investigation-20261108T174201Z/reproduction.json
```

Exit 0; seven checks passed. Results and stdout:
`artifacts/modal-jwt-investigation-20261108T174201Z/reproduction.json` and
`reproduction.log`. Five installed SDK source hashes match their package
RECORD entries; this was unmodified Modal 1.6.3 on Python 3.13.9.

Confirmed defect: `_AuthTokenManager._try_refresh_token` catches a non-auth
refresh exception and returns the old token whenever one exists, without
rechecking expiry. In the synthetic case, a token expiring at virtual time
1100 is returned at virtual time 1101 after refresh raises a transient error,
instead of propagating that error. No real JWT, refresh outage, server validation
or GPU job was used in this reproduction.

Controls establish the boundaries:

- Without a cached token, fetch failure propagates.
- A still-valid cached token can safely survive a proactive refresh failure.
- Authentication denial propagates instead of falling back.
- A successful refresh replaces an expired token.
- Actual `_InputPlaneInvocation.run_function` refreshes expired header metadata
  between polling calls; the claim that a long `.remote()` simply never refreshes
  authentication is false.
- A synthetic `AttemptAwait` rejection propagates even with locally valid
  metadata. That check reproduces error handling, not a server-side cause.

### Historical correlation and required vendor evidence

Read-only, lead-authored queries obtained lifecycle/layout metadata for app
`ap-XsRx6dMlnPhtSbg3H7haXL`:

- Created: `2026-11-08T09:10:55.497248+00:00`.
- Stopped: `2026-11-08T09:28:33.279679+00:00` (includes failure cleanup, not an
  exact JWT-rejection timestamp).
- `diagnose_timing` function: `fu-TLXga06VoZAjM7XDA6uNER`.

`AppGetObjects` and the function's `FunctionCallList` returned
`UnimplementedError`; `AppGetLayout` succeeded. Saved responses:
`historical-app-metadata.json` and `historical-app-layout.json` in the investigation
directory. These were metadata reads, not job submissions.

To finish attribution, Modal needs to inspect the failed `AttemptAwait` before
that app stopped and report:

1. Whether validation rejected the header JWT or the attempt token.
2. Its expiry and server validation time, plus the request's `requested_at`
   timestamp (to discriminate clock skew). Raw token strings are not needed.
3. `AuthTokenGet` refresh timestamps/statuses preceding that request, and any
   transport retries that reused metadata.
4. If it was the attempt token, its mint/expiry policy for this invocation.
5. A request/trace correlation ID and the backend validation reason.

The expired-cache defect is a supported candidate, not a historical verdict.
No SDK monkey-patch, dependency upgrade, credential change, measurement retry,
GPU allocation or support message was performed. Durable spawn/poll and recovery
remain the verified operational path while historical attribution awaits the
discriminating vendor records.

### Disposition: historical cause paused, SDK defect tracked separately (2026-11-08)

User decision: keep the historical `20261108T091114Z` JWT rejection marked
unknown and non-blocking, and track the reproducible Modal 1.6.3 SDK
expired-cache fallback defect separately.

- Historical cause: paused, not solved. It is not a current validation or
  release blocker.
- SDK defect: open and reproduced offline (seven synthetic, real-SDK
  control-flow cases; see reproduction artifacts above). It is tracked
  locally only — no upstream issue has been filed or fixed, and no link to
  the historical incident is confirmed.
- Escalation triggers — either suffices: a JWT/authentication failure recurs,
  or the verified durable spawn/poll/reattach path is affected.
- The five vendor evidence questions in the previous section are retained for
  that possible future escalation. They are not an outstanding vendor request
  and are not a validation gate.

### GPU tool diagnostics matrix — initial historical review (2026-11-08)

Initial frozen summary: `artifacts/gpu-diagnostics-review-20261108T195500Z.json`
(31 rows; raw inputs immutable in the earlier receipts).

- Acceptance: **partial** — 29 honest cases passed, 1 failed; the seeded
  invalid-access control was detected (its own result correctly stays failed).
- Compute Sanitizer (run `20261108T192232Z`): all four checks pass on CUDA,
  Triton, TileLang and cuTile. CuTe DSL **memcheck failed** — 18 CUDA API
  errors (`cuda_api_version_findings`; API version 13041/13050 exceeds driver
  13030); its other three checks pass. Open compatibility finding; no
  suppression, pin or kernel change made.
- Invalid-access control (Triton memcheck): 6 errors detected for the actual
  4-byte invalid global write to the named test kernel — detection works.
- Nsight Systems: the 5 GPU captures originally failed on a missing importer;
  the fix was a bounded OS library addition (`libdw1`, installing
  `libdw1t64`/`libelf1t64` 0.190-1.1ubuntu0.1, user-approved) plus CPU-only
  recovery of the existing QDSTRM traces — no GPU recapture. All 5 now pass;
  each shows exactly one kernel in the captured range. No memcpy activity
  table is present; transfer timing is unavailable (null, not zero). Captured range/process only — not
  whole-GPU utilization.
- Nsight Compute: the 5 original captures failed on the GPU clock-lock
  requirement; rerun with explicit `--clock-control none` (no clock or
  privilege changes) — all 5 pass, each with 1 selected kernel and 562
  exported raw metric columns preserved as strings/units. These are
  diagnostics, not reward latencies; the kernels are 1x128 toy axpby kernels
  and no optimization claim is made.
- Original `20261108T192232Z` run GPU `e0ec297f-c080-41d7-d6cf-9da2a35d89ef`;
  Nsight Compute run `20261108T193605Z` GPU `e774783c-9885-d188-5045-b88ddc547021`.
- Diagnostics are opt-in and isolated from rewards; cases run as separate
  processes, not a sandbox. Raw launch statuses are preserved; `reviewed.json`
  and the frozen summary are the authoritative post-review records.
- Source snapshots: `artifacts/gpu-diagnostics-source-20261108T192125Z`,
  `...-20261108T193538Z`. Recovery launcher:
  `artifacts/nsys-recovery-20261108T194721Z/` (run `20261108T194923Z`).
- Final local regression gate: system Python **174 tests OK, 32 skipped**;
  venv **174 tests OK, 26 skipped**, including 12 captured-output decoder tests at that gate (13 after the fail-closed hardening below).
  Evidence: `artifacts/final-gate-20261108T195620Z/{system,venv}.log`.


## H200 sanitizer and Nsight fixes — 20261108T200118Z

The first diagnostic batch (`20261108T192232Z`) remains historical evidence:
19/20 honest sanitizer checks reported zero errors; CuTe DSL memcheck reported
18 CUDA API-version errors despite five correct outputs. The invalid-access
control was detected. Nsight Systems retained raw QDSTRM files but could not
load its importer, and Nsight Compute could not lock GPU clocks.

Three fixes/mitigations are now verified together:

- `requirements-gpu.txt` explicitly pins `cuda-bindings==13.3.0`. The former
  13.5.0 bindings requested APIs 13041/13050 while the driver reported 13030.
  The [13.3.0 package](https://pypi.org/project/cuda-bindings/13.3.0/)
  satisfies CuTe DSL 4.8.1's `cuda-bindings>=12.8` requirement. Compiler and
  tileiras 13.3 pins are unchanged. No sanitizer API-error suppression was added.
- The Modal image now installs `libdw1` (`cloud/modal_runtime.py`), which supplies
  the `libdw.so.1` that QdstrmImporter was missing. Fresh collection now automatically
  produces `.nsys-rep` and SQLite output. The frozen reviewer validates worker
  completion, the generated NVTX range and in-range GPU kernel activity.
- The existing `--clock-control none` Nsight Compute setting avoids clock-lock
  operations ([NVIDIA CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)).
  Reviewed reports retain selected kernel identity, raw counter values and units.
  Unlocked-clock diagnostics are not reward latencies.

Sources were frozen before GPU execution, including concurrent Nsight reviewer
work; the source manifest checks every copied byte. Exact source and evidence:
[toolchain-fix bundle](artifacts/h200-toolchain-fix-20261108T195717Z/), including
`source-snapshot.json`, `source.tar.gz`, both GPU runs and raw local logs.
Protected models, bindings, stock CUDA kernels and upstream SKILL.md were not edited.

Pilot command (executed from the frozen source with the repository Modal venv):

```bash
modal run --detach cloud/modal_runtime.py --diagnose-kernels --backends cutedsl --diagnostic-tools sanitizer --diagnostic-checks memcheck --diagnostic-timeout-s 600
```

Run `20261108T195956Z`: installed bindings 13.3.0; CuTe DSL completed five
output checks with `ERROR SUMMARY: 0 errors` and tool exit 0. The separate
Triton invalid-access control reported six errors and was detected. No retry
or error suppression was used.

Full command:

```bash
modal run --detach cloud/modal_runtime.py --diagnose-kernels --diagnostic-timeout-s 600
```

Run `20261108T200118Z`: one diagnostic stage exited 0 without timeout in
283.120254639 s. Re-review with the exact frozen reviewer reports `pass`, all
31 planned cases present, 30 honest cases passed and the intentional control
correctly rejected. Every honest sanitizer case completed five output checks,
exited 0, and reported zero errors. Racecheck also reported zero hazards/warnings.

| Check | CUDA | Triton | TileLang | cuTile | CuTe DSL |
|---|---|---|---|---|---|
| memcheck | clean | clean | clean | clean | clean |
| racecheck | clean | clean | clean | clean | clean |
| synccheck | clean | clean | clean | clean | clean |
| initcheck | clean | clean | clean | clean | clean |
| Nsight Systems import + timeline review | pass | pass | pass | pass | pass |
| Nsight Compute counters, clock control none | pass | pass | pass | pass | pass |

Each Nsight Systems capture contains one in-range generated kernel and real,
nonempty report/SQLite files. Each Nsight Compute capture contains one selected
kernel and required finite counter metrics. A GPU-health XID 43 warning appeared
during the intentional invalid-access-control phase and is preserved in the
raw session log; all ten subsequent Nsight checks passed on that GPU. The warning
was not erased or treated as sanitizer-clean evidence.

Local commands on the frozen source:

```text
python3 -m unittest discover -v
Ran 174 tests in 17.147s
OK (skipped=32)

.venv/bin/python -m unittest discover -v
Ran 174 tests in 455.035s
OK (skipped=26)
```

The first local snapshot check lacked historical offline tracking fixtures.
Those data files were copied into its test bundle and the suites reran; no test
logic was weakened. Artifacts remain excluded from the cloud image except the
upstream checksum manifest. Diagnostic results are standalone `diag/*` evidence:
no reward input, result-schema field, timing method or tolerance changed.

- Post-review hardening: the sanitizer "clean" verdict now requires the log to
  contain *only* the exact expected vendor banner lines (plus an exact
  zero-error/zero-hazard summary); any unrecognized vendor diagnostic line
  fails closed to `unavailable`/`unrecognized_sanitizer_output` instead of
  passing. Applied to the final `sanitizer_clean` branch only. Targeted gate:
  `python3 -m unittest -v tests.test_diagnostic_review tests.test_diagnose
  tests.test_modal_diagnostics` — **37 tests OK** on both system Python and
  the venv (logs in `artifacts/hardening-gate-20261108T201923Z/targeted.log`).
- Closeout evidence check for `20261108T200118Z`: all 136 manifest entries match
  both the frozen source directory and source archive. Reviewed capture/raw-output
  hashes match the saved evidence; capture-harness hashes match the source
  snapshot. All 31 planned cases are present, all honest tool
  stages exited zero, and all 20 clean sanitizer outputs also satisfy the stricter
  post-review guard. No GPU measurement or timing reduction was rerun for this
  check. Record: `artifacts/h200-toolchain-fix-20261108T195717Z/lead-evidence-check.json`.

## H200 toolchain scoring regression — 20261108T200653Z

Full scoring acceptance with the `cuda-bindings==13.3.0` pin, run from the
same frozen source as the diagnostic matrix
(`PYTHONPATH=<frozen source> modal run --detach cloud/modal_runtime.py --tests`).
The scoring and runtime code in that snapshot matches the committed code. Only
documentation and the offline reviewer hardening above changed after the
freeze.

The automatic review reports `pass`, with no failures, no timing anomalies and
an environment matching the fresh require probe (driver 610.57.04, Torch
2.14.1+cu130, CuTe DSL 4.8.1, `runtime/cuda_bindings_version` 13.3.0). The
GPU suite reported `Ran 31 tests in 501.225s` / `OK`. All 11 profiled rollouts
record `profile/timing_method=cuda-leaf-sum-two-empty-prime-v1`; the
`timeout` and `max_turns` cases are rejected before profiling.

| case | provenance | eager µs | compile µs | generated µs | reward | reason |
|---|---|---|---|---|---|---|
| triton | 1.0 | 1.866 | 0.829 | 0.832 | 2.0 | faster_than_eager |
| overlong | 1.0 | 1.872 | 0.826 | 0.826 | 1.25 | faster_than_eager |
| cuda | 1.0 | 1.853 | 0.819 | 0.964 | 2.0 | faster_than_eager |
| tilelang | 1.0 | 1.862 | 0.864 | 0.810 | 3.0 | faster_than_both |
| cutile | 1.0 | 1.862 | 0.819 | 0.922 | 2.0 | faster_than_eager |
| cutedsl | 1.0 | 1.866 | 0.829 | 0.976 | 2.0 | faster_than_eager |
| reward-hack-torchop (5 backends) | 0.0 | 1.839–1.856 | 0.785–0.834 | 1.760–1.837 | -1 | provenance |
| timeout / max_turns | — | — | — | — | -1 | timeout / max_turns |

Eager medians here (1.84–1.87 µs) are higher than in unprimed historical runs
(about 1.60–1.66 µs in the generated-first order). That is consistent with the
primer removing the order-dependent speedup, but these runs are not
equivalent measurements and no historical timing was regraded.

TileLang's `faster_than_both` comes from a 0.810 vs 0.864 µs margin at
sub-microsecond size. Read it as a single-run fixture result, not a
performance claim.

A re-review of the saved matrix `20261108T200118Z` with the hardened reviewer
(run on a copy, so the saved evidence is untouched) still reports `pass`: 30
honest cases passed and the invalid-access control was rejected.

Evidence: [scoring run](artifacts/h200-toolchain-fix-20261108T195717Z/scoring-20261108T200653Z/)
(acceptance review, results, kernel names, environment, stage logs and
benchmark samples).

## Step 2: diagnostic profiling artifacts (2026-11-08, CPU only)

`--diagnostics` adds the `utils.profile_artifacts` stage, which writes
`<workdir>/.artifacts/<run_id>/{report,kernels,trace}.json`. Every result also
carries `env/provenance_{agent,library,torch,memory}` now; the default key set
is 22 keys, pinned explicitly as `tests.support.DEFAULT_RESULT_KEYS`. With
diagnostics off, the stage argv and reward are unchanged. **Nothing in this
section ran on a GPU.** The GPU capture path is covered only by the
skip-marked `tests.test_profile_artifacts.GPUArtifactTests`.

```bash
# WSL2, inside AgenticRL-CUDA-kernels/
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover
# Ran 189 tests in 92.726s
# OK (skipped=33)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover
# Ran 189 tests in 304.645s
# OK (skipped=27)

# End-to-end CPU rollout of a fresh Triton task copy (tests.support.make_task)
CUDA_AGENT_DEVICE=cpu TRITON_INTERPRET=1 .venv/bin/python -m utils.rollout_env     /tmp/step2-cpu/triton --diagnostics --benchmark --timeout-s 600
# env/verify_ok true, reward/reason "unmeasured", env/provenance_* null,
# artifact/report and artifact/kernels written, artifact/trace null,
# artifact/unavailable_reason "CUDA_AGENT_DEVICE=cpu", bench/available false
```

The CPU report records `gpu: null` with `gpu_reason: CUDA_AGENT_DEVICE=cpu`,
`trace: null` (no `trace.json` is written), `capture: null`,
`provenance: null`, input shapes `[1, 128]` float32 ×2, Torch 2.14.1+cpu,
Triton 3.8.0, `driver: null` (nvidia-smi not found), 34 source hashes, every
external tool unavailable on PATH, and CUPTI unavailable. `kernels.json` has
`measured: false` and no rows. The embedded benchmark record is the CPU
unavailable record. Evidence: [step2-tests-20261108](artifacts/step2-tests-20261108/)
(both suite logs, the rollout result and the copied artifact directory).
The GPU path was validated afterwards in H200 run `20261108T225150Z` (next
section).

Synthetic, Torch-free tests cover: kernel-table aggregation and provenance
classes (memory, unknown-as-torch); strict-JSON report round trips; source
hashes changing only for the edited kernel file; tool and CUPTI capability
states; and diagnostics-stage failures (exit, timeout, malformed output, a
child claiming a path it did not write). None of those failures changes the
reward, and the parent reports only files present in its own artifact
directory.

## Step 2 on H200: run 20261108T225150Z

Full acceptance, with `tests.test_profile_artifacts.GPUArtifactTests` added to
the Modal GPU targets. The runner sets `CUDA_AGENT_KEEP_PROFILE_ARTIFACTS`, so
that test copies its artifact directory into the run's volume folder before
scratch cleanup.

```bash
# WSL2, inside AgenticRL-CUDA-kernels/
.venv/bin/modal run --detach cloud/modal_runtime.py --tests
# app ap-O8LWmk6MPfpSs9dE9rEKWi, run_id 20261108T225150Z, NVIDIA H200,
# driver 610.57.04, Torch 2.14.1+cu130, CUDA runtime 13.0, Triton 3.8.0,
# cuda-bindings 13.3.0
# gpu_tests: Ran 32 tests in 669.379s / OK
#   (31 previous GPU tests + test_trace_kernel_table_and_report ... ok)
# acceptance: pass (no failures, no timing anomalies)
```

**Source match.** Before launch, the SHA-256 of all 67 `.py` files under
`utils/`, `cloud/`, `tests/`, `scripts/`, `kernels/` and `tracking/` was
recorded (`local-source-sha256.txt`); none changed before the evidence was
committed. All 21 `harness/utils/*` hashes in the run's `report.json` match the
committed files byte for byte. The run used the code committed with this
evidence: Step 2 plus the Modal target and evidence-copy change.

**Diagnostic artifacts** (`profile-artifacts/20261108T230513Z-9d48f02a/`, the
Triton fixture run with `diagnostics=True, benchmark=True`):

- `report.json`: GPU NVIDIA H200, compute capability 9.0, 150101688320 bytes,
  UUID recorded; every tool on PATH (`nvidia-smi`, `nvcc`, `ninja`, `nsys`,
  `ncu`, `compute-sanitizer`) and CUPTI available with 10 CUDA events; capture
  of 10 CUDA and 43 CPU events, no failure reason, primer counts `[0, 0]`;
  benchmark samples embedded (`bench/available: true`).
- `kernels.json`: `measured: true`, one row, `axpby_kernel`, 10 calls,
  8.320 µs total, 0.832 µs mean, class `agent`; provenance share agent 1.0.
- `trace.json`: Chrome trace with 146 events.
- `rollout-result.json`: `artifact/unavailable_reason` null; reward 2.0
  (`faster_than_eager`), the same as an ordinary rollout. The diagnostic pass
  did not change it.

These diagnostic timings are instrumented (CPU activity, shapes, memory) and
are not reward inputs.

**Scoring** (`scripts.validate`, primed method; the reward still uses `t_*_us`):

| case | provenance | eager µs | compile µs | generated µs | reward | reason |
|---|---|---|---|---|---|---|
| triton | 1.0 | 1.802 | 0.800 | 0.845 | 2.0 | faster_than_eager |
| overlong | 1.0 | 1.812 | 0.803 | 0.829 | 1.25 | faster_than_eager |
| cuda | 1.0 | 1.799 | 0.800 | 0.966 | 2.0 | faster_than_eager |
| tilelang | 1.0 | 1.801 | 0.800 | 0.813 | 2.0 | faster_than_eager |
| cutile | 1.0 | 1.796 | 0.800 | 0.890 | 2.0 | faster_than_eager |
| cutedsl | 1.0 | 1.788 | 0.797 | 0.998 | 2.0 | faster_than_eager |
| reward-hack-torchop (5 backends) | 0.0 | 1.795–1.815 | 0.791–0.841 | 1.824–1.837 | -1 | provenance |
| timeout / max_turns | — | — | — | — | -1 | timeout / max_turns |

The new flat `env/provenance_*` keys appear in every profiled case and agree
with `env/provenance` (agent 1.0 for honest fixtures). TileLang scored
`faster_than_both` in `20261108T200653Z` (0.810 vs 0.864 µs) and
`faster_than_eager` here (0.813 vs 0.800 µs). That is the sub-µs reward noise
already listed in KNOWN_ISSUES, not a regression.

Scope: one H200, the shipped 1 × 128 fixtures, and the diagnostic artifact
pass for Triton only. Kernel classes are name-based heuristics.

Evidence: [step2-h200-20261108T225125Z](artifacts/step2-h200-20261108T225125Z/)
(session log, pre-launch checksums, and the run folder with the acceptance
review, results, kernel names, environment, stage logs, benchmark samples and
profile artifacts).

## Progressive task suite (2026-11-09)

`tasks/` holds 20 workloads in stages 0–4 (README, "Kernel tasks the agent
works on"). They are evaluated by `python -m scripts.validate --suite ...`
(`utils/task_suite.py`) and on Modal by `cloud/modal_tasks.py`, from a frozen
source tree (`scripts/freeze_task_suite.py`). Two task-only gates were
approved by the user on 2026-11-09: the `authored-only-no-library-v1`
provenance gate (`utils/task_policy.py`) and exact integer-output comparison
(`RolloutConfig.exact_integer_outputs`, `utils/task_verification.py`). The
stock rollout keeps its reward and tolerances.

**H200, stages 0–1: run `task-stage1-20261109T004826Z`.** Smoke case, CUDA
and Triton, honest/hack/wrong fixtures: status `pass`, 24 passed, 0 failed,
0 unavailable. Every honest solution scored (`correct` to
`faster_than_both`), every Torch-op hack failed `provenance`, and every
wrong-output control failed `verification`. The frozen source (175 files) is
in `artifacts/task-suite-source-20261109T002404Z-stage1/`, and the evidence is
in `artifacts/task-stage1-20261109T004826Z/`. After that run, `utils/task_policy.py`,
`utils/task_suite.py`, `utils/task_fixtures.py`, `utils/verification.py`,
`utils/config.py` and `utils/rollout_env.py` changed (cuBLAS ceiling record
validation, candidate source hashes, exact integer mode, fixture generator).
The stage 0–1 kernels and task files did not. The run does not cover those
harness changes.

**CPU interpreter.** Codex's manifests under `artifacts/task-stage1-20261108T234354Z/`,
`task-stage2-*`, `task-stage4-*` and `task-gemm-tc-fix-*` record per-task
verification. Stages 0–3 pass 5/5, except `gemm_tensor_core` (the interpreter
mis-evaluates bf16 `tl.dot`). In stage 4, only `decode_attention` passes.
`flash_attention_fwd`, `grouped_gemm` and `moe_dense_gemm` fail under the
interpreter's bf16 `tl.dot`; `residual_rmsnorm` has 35 of 131072 elements out
of tolerance; and `topk_sampling` did not finish. None of those causes is
confirmed.

**H200, stages 2–4 and larger cases.** Each run used a frozen source
(`task-source-manifest.json` in its evidence):

| Run | Scope | Result |
|---|---|---|
| `task-stage23-20261109T013032Z` | stages 2–3, smoke, CUDA + Triton | 51 passed, 3 failed: `gemm_tensor_core` CUDA did not compile (the `(T)0` padding under `__CUDA_NO_HALF_CONVERSIONS__`), fixed in `task-gemm-tc-fix-20261109T013750Z` |
| `task-stage4-smoke-20261109T020100Z` | stage 4, smoke | 33 passed, 3 unavailable (`grouped_gemm` CUDA is not provided by contract) |
| `task-flagship-acceptance-20261109T021131Z` | `gemm_tensor_core` + `flash_attention_fwd` acceptance | 12 passed; Triton GEMM 4096³ bf16 at 0.854 of cuBLAS (gate ≥ 0.6) |
| `task-cute-retest-20261109T021131Z` | `gemm_tensor_core` CuTe DSL, smoke | fail: CuTe DSL compile error |
| `task-decode-long-…`, `task-flash-middle-…`, `task-grouped-large-…`, `task-grpo-nontrivial-…`, `task-layernorm-bf16-…`, `task-moe-shared-…`, `task-packed-sum-…`, `task-residual-large-…`, `task-sampling-k32-…`, `task-scan-large-exact-…` (all `-20261109T021131Z`) | one larger case each | all pass (3–6 rows each) |

Source match against the committed tree. The stage 2–4 smoke runs and the
larger-case batch differ from it only in `utils/task_suite.py` (a second
protected-file check after the cuBLAS ceiling stage, which can only turn a
pass into a fail), `tests/test_task_measurement.py`, and
`tasks/gemm_tensor_core/kernels/cutedsl/gemm_tensor_core.py` (rewritten after
the CuTe DSL failure; not GPU-tested). The stage 2–3 run also predates the
CUDA GEMM fix, and its `scripts/task_readme.py`/`tests/test_task_sampling.py`
differ. For the larger-case batch only the durable results and launch
receipts were saved, not the full per-case logs.

**Test suites** (WSL, inside `AgenticRL-CUDA-kernels/`; logs in
`artifacts/task-suite-tests-20261109/`):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover
# Ran 233 tests in 119.145s
# OK (skipped=46)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover
# Ran 238 tests in 437.862s
# OK (skipped=27)
```

### CuTe DSL GEMM fix: runs task-cute-fix-{smoke,acceptance}-20261109T030234Z

The retest `task-cute-retest-20261109T021131Z` failed IR verification:
`'cute.copy' op expects pred to have compatible shape with: (2,(2,4)) but got
actual predShape: ((2,2),(2,4))`. The C store was a predicated `cute.copy`
whose predicate was built from the MMA output-fragment shape. The fix
(already in the committed tree at `51e71b4`) stores each accumulator element
under an identity-tensor bounds check, with no copy predicate. The A/B tile
copies are unchanged.

```bash
# WSL2: freeze the committed tree, then run from the frozen copy
.venv/bin/python -m scripts.freeze_task_suite --tasks gemm_tensor_core     --output /tmp/task-cute-fix-source-20261109T030234Z
cd /tmp/task-cute-fix-source-20261109T030234Z && PYTHONPATH=$PWD   <repo>/.venv/bin/modal run cloud/modal_tasks.py --tasks gemm_tensor_core   --case-set {smoke|acceptance} --backends cutedsl --fixtures honest,hack,wrong   --run-id task-cute-fix-{smoke|acceptance}-20261109T030234Z
```

Both runs: status `pass`, 3/3. The uploaded source matches the frozen
manifest, and all 136 frozen files match the committed tree (git `51e71b4`).

| Case | Fixture | Result | Generated µs | Eager µs | Compile µs | cuBLAS fraction |
|---|---|---|---|---|---|---|
| 512³ bf16 | honest | `correct`, authored provenance 1.0 | 60.8 | 20.0 | 18.8 | 0.143 |
| 512³ bf16 | hack / wrong | `provenance` / `verification` (expected) | — | — | — | — |
| 4096³ bf16 | honest | `correct`, authored provenance 1.0 | 7068.9 | 2886.6 | 2724.3 | 0.026 (19.4 TFLOP/s) |
| 4096³ bf16 | hack / wrong | `provenance` / `verification` (expected) | — | — | — | — |

The CuTe DSL kernel is now correct but slow: about 2.5× slower than eager at
4096³. It uses universal (non-async) 16-bit copies, a single shared-memory
stage and a 64 × 64 × 32 tile. The ≥ 60% cuBLAS acceptance gate applies only
to the Triton backend, so these rows pass. Evidence:
`artifacts/task-cute-fix-20261109T030234Z/`.

## Stage 5 EAGLE-4 evaluation workloads (2026-11-09)

Source/license review preceded implementation. The actual documentation, root
Eagle 4 Source-Available License, incorporated trademark policy, NOTICE,
third-party notices and LICENSES texts are pinned at upstream commit
`14fb79a5a4562ebe671444d7841253b889301040` in `legal/eagle4/`; the manifest
records exact URLs, SHA-256 and retrieval time. Section 2 permits the requested
noncommercial evaluation/benchmark adaptations subject to the license. These
files are source-available, not OSI-approved open source. Attribution, source,
commercial/competitive and notice obligations remain. All nine tasks are
`eval_only=True`; this implementation does not clear training use.

The complete contracts, license acceptance/attribution notices, source deviations,
case grids, fixture variants and accounting conventions are in `EAGLE4_STAGE5.md`.
No eagle_harness implementation source was fetched or ported. The documented
batch-1 greedy walk was extended locally to B=8/32 and sampled acceptance; Mamba2
conv uses the common width-4 tree windows, without selective state update. The
GDN walk/chain replay use the existing floating tolerance, not bitwise state
comparison. No RolloutConfig fields, reward semantics, tolerances, stock upstream
files or dependencies changed.

Added nine self-contained `tasks/eagle_*/` directories (Torch reference, authored
Triton and CUDA sources/bindings, Torch-op and wrong-output controls, plus
wrong-ancestor controls for all eight tree tasks). Recurrence also has a Triton
`model_new_chain_replay.py`. Added `utils/eagle_contracts.py`,
`utils/eagle_reference.py`, `scripts/verify_stage5_cpu.py`, regression tests,
source/legal preservation and stage/fixture support in the existing catalog,
input dispatcher, suite and source freezer. The expanded catalog has 3,618 Stage
5 cases; it is not a claim of 3,618 tested cases.

Local runtime: WSL2, Python 3.13.9, Torch 2.14.1+cpu, Triton 3.8.0, NumPy 2.5.4;
CUDA unavailable, nvcc absent. Raw versions: `artifacts/task-stage5-20261109T084046Z/runtime-cpu.json`.

### Strict CPU interpreter smoke checks

The first pass used the exact subprocess argv and environment in
`artifacts/task-stage5-20261109T084046Z/manifest.json`, with actual copied
candidates under each task/fixture directory and raw `verification.log` files.
Its `verify_stage5.py` driver is preserved there. Each subprocess ran
`utils.task_verification` (stock verification plus exact integer comparisons)
through a scaffolded workdir with `CUDA_AGENT_DEVICE=cpu`, `TRITON_INTERPRET=1`,
`PYTHONDONTWRITEBYTECODE=1`, OMP/MKL threads=2 and a 600-second timeout.
The initial harness was not a pre-frozen tree; that manifest explicitly separates
end-of-run repository hashes from the copied candidates that actually executed.

| Task/fixture | Smoke CPU result |
|---|---|
| eagle_candidate_topk / honest | **unverified**: 600-second interpreter timeout; no pass claimed |
| eagle_tree_step / honest | 5/5 after exposing exact selected flat child IDs; final-fixes evidence below |
| eagle_tree_row_attention / honest | 5/5 |
| eagle_tree_verify_attention / honest | 5/5 |
| eagle_gdn_tree_conv / honest | 5/5 |
| eagle_mamba2_tree_conv / honest | 5/5 |
| eagle_gdn_tree_recurrence / honest | 5/5 |
| eagle_gdn_tree_recurrence / chain_replay | 5/5 |
| eagle_accept_tail / honest | 5/5 |
| eagle_accept_tail_sampled / honest | 5/5 after fixing the store mask to Boolean; final-fixes evidence below |

The first sampled run failed on an integer-valued tl.store mask. Its raw failure
is retained, rather than labeled interpreter-unsupported. The fix is shared by
CPU and GPU paths. The subsequent mask-fix smoke evidence is
`artifacts/task-stage5-sampled-fix-20261109/`. A final code review also made CHILD
a runtime argument to avoid recompiling the rejection kernels for every child,
and added selected flat child IDs to tree-step outputs so ties are observable.
Both affected tasks were rechecked against their final sources:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.verify_stage5_cpu \
  --tasks eagle_tree_step,eagle_accept_tail_sampled --fixtures honest \
  --output artifacts/task-stage5-final-fixes-20261109
# Both rows verified: 5/5 checks each; manifests include pre-run source/harness hashes.
```

The logs are under `artifacts/task-stage5-final-fixes-20261109/` and the launcher
stdout is `artifacts/stage5-final-fixes.log`. The 600-second top-k timeout was not
worked around by substituting a CPU implementation, changing precision, shrinking
the catalog smoke case or loosening comparison. Small **synthetic** random/tie
selection probes (including a shard boundary) pass separately; they do not
validate the full catalog top-k case.

### Branching, controls and offline compilation

`tests/eagle_interpreter_probe.py`, executed by the virtualenv test suite in an
isolated CPU interpreter subprocess, compares the authored tree kernels against
references for N=1/8, B=2 and chain/wide/balanced/ragged parents. It compares GDN
walk and chain replay, and checks top-k random scores and stable ties at small
synthetic vocabularies. Exact parent/window/path metadata checks are included.
The final log is `artifacts/task-stage5-20261109T084046Z/branching-probes-final.log`.
Reference tests additionally cover ties, full windows, sibling exclusion,
request-history windows, Mamba channel layout, parent-versus-previous-node state,
rejection residual correction, seeded reproducibility, exact large slot IDs,
Torch-op controls verifying, both wrong controls failing, and logical-byte counts
against actual input/output tensors.

Offline compilation used meta tensors and an explicit CUDA sm90/warp32 compiler
target; it never initialized a GPU execution context or launched a kernel:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  artifacts/task-stage5-20261109T084046Z/compile_eagle_triton_final.py
# COMPILATION_DONE 48 0
```

All 48 distinct smoke/largest-shape Triton configurations compiled. The raw log
is `artifacts/stage5-offline-compile-final.log`; source signatures, constexprs,
PTX hashes and compiler results are in
`artifacts/task-stage5-20261109T084046Z/offline-triton-final.json`.
The earlier 237-configuration compile (before removing child-index specialization)
is retained separately and is not the final-source gate. Compilation alone does
not verify runtime resource use, numerical output, timing, CUDA C++ builds or
provenance.

### Final local suites and remaining limits

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  > artifacts/stage5-system-tests-final.log 2>&1
# Ran 249 tests in 236.617s
# OK (skipped=56)
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  STAGE5_PROBE_OUTPUT=artifacts/task-stage5-20261109T084046Z/branching-probes-final.log \
  .venv/bin/python -m unittest discover > artifacts/stage5-venv-tests-final.log 2>&1
# Ran 254 tests in 943.844s
# OK (skipped=28)
```

Only the final logs above gate the final output contract. Earlier exploratory
logs are retained; an earlier concurrent suite observed the tree-step arity/byte
contract while it was being updated and is not final-source acceptance.

The system suite skips Torch-dependent references/interpreter checks and GPU or
missing optional-package checks. The virtualenv runs the new reference and
interpreter checks; its new GPU test explicitly skips because CUDA is unavailable.
No GPU fixture has been accepted, no GPU timing/bandwidth/TFLOP/s is reported, and
no paid Modal invocation or live W&B/Weave write was made.

Open verification work: full-size top-k correctness, all CUDA C++ builds, device
numeric behavior/provenance, and larger catalog cases. The CUDA sources are
conservative correctness fixtures, not performance claims. Source freezes retain
all legal notices. A GPU smoke run and then one selected larger case per task
need separate user approval; no new tolerance or RolloutConfig decision is needed.

Final CUDA-source review flattened the candidate vocabulary launch into grid.x,
so the 128256/151936 vocabulary cases do not use the limited grid.y dimension.
This CUDA-only index/launch change followed the broad suites; Python/reference
contracts were unchanged. CUDA remains uncompiled and unexecuted locally. The
applicable catalog, measurement and integer-verification regression checks pass:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_task_catalog \
  tests.test_task_measurement tests.test_task_integer_verification \
  > artifacts/stage5-final-cuda-source-regression.log 2>&1
# Ran 31 tests in 23.135s
# OK (skipped=12)
```

### Stage 5 on H200 (2026-11-09)

Both runs used one frozen source, `/tmp/task-stage5-gpu-source-20261109T190928Z`
(285 files, manifest SHA-256 `b57a4ace261736d5eeecb480922388b95eff58abe13da67c7c8d172aca5c1169`,
git `166bf02`).

```bash
cd /tmp/task-stage5-gpu-source-20261109T190928Z && PYTHONPATH=$PWD   <repo>/.venv/bin/modal run [--detach] cloud/modal_tasks.py --tasks stage5   --case-set {smoke|acceptance} --backends cuda,triton --fixtures all --timeout-s 900
```

**Phase 1, smoke (`task-stage5-gpu-smoke-20261109T190928Z`):** status
`pass`, 71 passed, 0 failed, 0 unavailable, no source mismatches, 2637 s.
These were the first Stage 5 CUDA C++ builds. Honest and chain-replay
solutions scored; hacks failed `provenance`; wrong and wrong-ancestor
controls failed `verification`. The local client was killed by host memory
pressure mid-run, but the remote job completed and wrote its summary.

**Phase 2, largest cases (`task-stage5-gpu-acceptance-20261109T190928Z`,
detached):** stopped on purpose after 14 rows (`progress-stopped.jsonl`).
- `eagle_candidate_topk` (32 × 151936 vocabulary): CUDA passes all three
  fixtures. Triton honest fails `Triton Error [CUDA]: invalid argument`:
  `eagle_vocab_score[(b, v)]` puts 151936 programs on grid axis 1, whose limit
  is 65535. Fixed by launching `(v, b)` and swapping the program ids.
- `eagle_tree_step` (32 × 64 ragged): honest fails at profiling on both
  backends (`JSONDecodeError: Extra data`), and the hack times out (901 s);
  the wrong and wrong-ancestor controls pass. Cause: the trusted references in
  `utils/eagle_reference.py` walked trees with per-element `int(tensor)` host
  syncs. The eager baseline then produced so many profiler events that its
  single JSON record exceeded the harness's 4 MiB stdout tail, and the hack
  (which calls the reference) ran out of time. Fixed by vectorizing
  `ancestors`, `tree_conv`, `recurrence`, `accept` and `accept_sampled`
  across the batch, with no host syncs.
  `tests/test_eagle_reference_vectorized.py` checks them against the
  original loops (frozen in `tests/fixtures/eagle_reference_loop.py`), and
  every output is bit-for-bit equal (7 tests OK). The eager baseline for
  these tasks is now a real GPU reference, so "faster than eager" means more.

Local suites after the fixes (`artifacts/runs/task-stage5-reference-fix-20261109/`):
system Python `Ran 262 tests in 348.337s` / `OK (skipped=63)`. The venv suite
was killed by host memory pressure before finishing, with no failures up to
that point. The equivalence tests passed in the venv on their own.

### Stage 5 largest cases, rerun: task-stage5-gpu-acceptance-20261109T222837Z

Detached H200 run of the fixed tree (frozen source: 289 files, manifest
`53dd812bd58f92a38c957de7939750e4bfe991cb8a03e572e200035328df7a1c`, git
`86441c8`): status `fail`, 52 passed, 19 failed, no source mismatches,
4417 s. Evidence: `artifacts/runs/task-stage5-gpu-acceptance-20261109T222837Z/`.

- **Both earlier fixes held on the GPU.** Triton candidate top-k passes at
  32 × 151936, and `eagle_tree_step` passes on both backends.
- Also passing at the largest case: `eagle_gdn_tree_conv`,
  `eagle_mamba2_tree_conv`, row-attention honest, and every wrong and
  wrong-ancestor control.
- **2 failures: verify-attention honest, illegal memory access** (CUDA at
  verification, Triton at compile). The partial-output offsets were computed
  in int32: (b·hq·n·splits)·d ≈ 4.3e9 > 2^31 at a 32768-token context. Row
  attention passed only because its 4096 window gives 33 splits instead of
  257. Fixed with int64 offsets in all four attention kernels (CUDA and
  Triton, row and verify).
- **17 failures: profiler record overflow** (`JSONDecodeError`): the
  attention hacks, and every honest/hack/chain-replay row of the GDN
  recurrence, greedy acceptance and sampled acceptance. Their references are
  sequential over 64 nodes and legitimately launch hundreds of kernels per
  forward. `profile/<session>_kernel_samples` listed every event in the
  single JSON line, which exceeded run_task's 4 MiB stdout tail. Fixed (user
  approved, 2026-11-09): the record now carries per-kernel aggregates
  `[name, calls, total_us]`, and the verbatim per-event lists go to a sidecar
  JSON whose path is `profile/kernel_samples_path`. Timings, provenance and
  rewards are computed as before. `ancestors()` now uses binary lifting
  (about 6 doubling steps instead of 64 depth steps), still bit-for-bit equal
  to the original loop.

Local suites after these fixes: system Python `Ran 264 tests in 352.512s` /
`OK (skipped=63)`. Earlier, after the first two fixes, the venv suite ran
`Ran 267 tests in 664.506s` / `OK (skipped=28)`.
