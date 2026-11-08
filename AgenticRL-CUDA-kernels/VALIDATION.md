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
