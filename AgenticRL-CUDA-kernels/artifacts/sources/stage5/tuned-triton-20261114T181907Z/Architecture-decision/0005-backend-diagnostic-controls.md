# ADR 0005 — Diagnostic controls for the cuTile and CuTe DSL backends

- **Status:** recorded (2026-11-13)
- **Scope:** compiler and IR inspection controls for the cuTile and CuTe DSL backends, and which of them reach a Modal job. Diagnostic only: no control here changes compilation for scoring, a timing, a gate or a reward.
- **Pins:** cuTile `cuda-tile==1.8.0` with `nvidia-cuda-tileiras==13.3.36`; CuTe DSL 4.8.1 (`requirements-gpu.txt`, `requirements-cutile.txt`, `requirements-cutedsl.txt`).

## Decision

Each DSL is inspected with its own controls, never Triton's. Triton's `MLIR_ENABLE_DUMP`
and `TRITON_KERNEL_DUMP` do not apply to either DSL; on a DSL run they can only dump the
Inductor baseline. A Modal job receives a control only through the `DIAGNOSTIC_ENV`
allowlist in `cloud/modal_runtime.py` (`forwarded_environment`), and only when diagnostics
are enabled; nothing else in the caller's environment is forwarded
(`tests/test_modal_diagnostics.py`). Compilation caches are always task-local, set by the
harness through each backend's `cache_env` in `utils/backend.py`.

| Backend (pin) | Control | What it does | Status in this repo |
|---|---|---|---|
| cuTile 1.8.0, tileiras 13.3.36 | `CUDA_TILE_LOGS=CUTILEIR` | Prints the Python frontend IR | Works; forwarded |
| | `CUDA_TILE_DUMP_BYTECODE=<dir>` | Saves TileIR bytecode | Works; forwarded |
| | `CUDA_TILE_LOGS=TILEIR`, `CUDA_TILE_DUMP_TILEIR` | TileIR text logs and dumps | Need CUDA Toolkit 13.4+; unavailable under the 13.3 pin, not forwarded, not recommended |
| | `CUDA_TILE_CACHE_DIR` | Compilation cache location | Set per task by the harness |
| | `cutile_cache`, `cutile-cache` | Inspect the compilation cache | Available in the package |
| CuTe DSL 4.8.1 | `CUTE_DSL_PRINT_IR=1` | Prints the generated MLIR | Forwarded |
| | `CUTE_DSL_KEEP=ir,ptx` with `CUTE_DSL_DUMP_DIR=<dir>` | Keeps IR and PTX files | Forwarded |
| | `CUTE_DSL_LINEINFO=1` | Line info to map profiler output to Python source | Recommended in `SKILL_CUTEDSL.md`; **not in `DIAGNOSTIC_ENV`**, so not forwarded yet |
| | `CUTE_DSL_CACHE_DIR` | Compilation cache location | Set per task by the harness |

Use `<dir>` under the remote artifacts volume (for example `/artifacts/<run_id>/cutile-ir`
or `/artifacts/<run_id>/cutedsl-ir`) so dumps survive the job.

## Rules for using them

- **Fresh cache when dumping.** A compilation-cache hit can suppress compiler output; dump
  from a fresh process with an empty task cache.
- **Diagnostics stay out of scoring.** A dump or print control may slow compilation; it
  is never set on a rollout whose timings feed a reward.
- **Version-gated controls stay off.** The TileIR controls wait for a CUDA Toolkit 13.4+ pin;
  adding them earlier would advertise controls the image cannot honour.
- **The allowlist is the boundary.** A new control is usable on Modal only after it is added
  to `DIAGNOSTIC_ENV` with a test, as `CUTE_DSL_LINEINFO` still needs to be.

## Sources

The names were checked against the pinned releases, recorded in
[VALIDATION.md](../VALIDATION.md) (backend inspection-controls table):
[cuTile 1.8.0 debugging](https://github.com/NVIDIA/cutile-python/blob/v1.8.0/docs/source/debugging.rst)
and [`_debug.py`](https://github.com/NVIDIA/cutile-python/blob/v1.8.0/src/cuda/tile/_debug.py);
[CuTe DSL 4.8.1 debugging](https://github.com/NVIDIA/cutlass/blob/v4.8.1/media/docs/pythonDSL/guides/debugging.rst).
The CUDA Toolkit 13.4 requirement for the TileIR controls is recorded in
`CUTILE_kernels_task.md`. Backend policy files: [SKILL_CUTILE.md](../SKILL_CUTILE.md),
[SKILL_CUTEDSL.md](../SKILL_CUTEDSL.md).
