# Nsight profiles and suite diagnostics: local evidence

Local evidence for `NSIGHT_profiling_task.md` (implemented 2026-11-14). **No GPU, vendor tool or Modal
job ran.** Nothing here is committed or launched.

## Files

| File | Contents |
|---|---|
| `torch-free-suite.log` | Full output of `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover` in WSL: `Ran 390 tests in 450.363s`, `OK (skipped=82)` |
| `prepared-requests.json` | Three unlaunched Modal requests (`launched: false`), below |

## The CPU-Torch suite did not finish

`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover` was stopped by Claude Code
because the machine ran low on memory; it produced no summary, so no result is recorded for it. The new
tests and the related modules passed under the venv separately (`tests.test_nsight_profiles`,
`test_diagnose`, `test_diagnostic_review`, `test_modal_diagnostics`, `test_gpu_lock`, `test_tracking`,
`test_task_catalog`: `Ran 111 tests`, OK), before one later fix to `cloud/modal_tasks.py` that the
Torch-free suite covers.

## Prepared requests

| Request | Tasks | Backends | Option | Honest rows | Estimate |
|---|---|---|---|---|---|
| Sanitizer A | 9 Stage 5 + 6 Stage 6 tasks | cuda, triton | `--sanitizer memcheck` | 30 | ~6,100 s (guess) |
| Sanitizer B | the nine first-round cuTile tasks | cutile | `--sanitizer memcheck` | 9 | ~1,900 s (guess) |
| ncu | four Stage 7 tasks | triton, cuda, cutile | `--ncu roofline` | 12 | ~1,200 s (guess) |

All use `--case-set smoke --fixtures honest` and need a frozen source, made after the changes are
committed; `source_manifest_sha256` is null until then. Estimates are guesses: none of these kernels
has run under these tools.
