# Stage 7 Tier 1: local evidence and the prepared H200 smoke request

Stage 7 holds four TileGym-derived tasks: `relu_squared`, `silu_and_mul`, `geglu_tanh` and
`rotary_embedding`. This folder records what was checked locally before any GPU run and the request
that would run them on an H200. **The request has not been launched.**

## Not verified on a GPU

Nothing in Stage 7 has GPU evidence. The CUDA kernels (`.cu` plus binding) have never been compiled.
The cuTile kernels have never run: cuTile has no CPU interpreter and is not in the local `.venv`. The
request below is the first test of both. The Triton kernels were checked only in the Triton CPU
interpreter, by the session that wrote them.

## Local test suites

Both full suites ran in WSL on the tree that became commit `a4a7e9c`, one at a time.

| Suite | Command | Result |
|---|---|---|
| Torch-free | `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover` | `Ran 353 tests in 918.843s`, `OK (skipped=82)` |
| CPU Torch + Triton interpreter | `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover` | `Ran 358 tests in 1527.144s`, `OK (skipped=29)` |

## Frozen source

`artifacts/sources/stage7/tier1-20261113T175914Z/`, written by
`.venv/bin/python -m scripts.freeze_task_suite --tasks stage7`. Its manifest records git revision
`a4a7e9cfcc0db0ddaf897199e79944517c0a698f` (the pushed HEAD at freeze time), 245 files, and has SHA-256
`abecc24e54c70d971e567d5705039117b65fc37d76b36a9f8029bd2b5f502742`.

## Request

`tier1-request.json`: run ID `stage7-tier1-smoke-20261113T175914Z`, `--tasks stage7 --case-set smoke
--backends triton,cuda,cutile --fixtures all --timeout-s 1200`, one H200.

- **Rows:** 4 tasks × 3 backends × 3 fixtures = 36.
- **Smoke cases:** `b64_h1024_float32` for the three activation tasks and
  `b1_hq32_hkv8_s512_d64_float32` for `rotary_embedding`. The smoke set is fp32 only; bf16 and the
  largest shapes need `--case-set acceptance` or `all`, which this request does not run.
- **Expected outcomes:** honest rows verify with an authored share of at least 0.9; hack rows fail
  provenance; wrong rows fail verification. A suite row "passes" when its fixture behaves as expected.
- **Estimated wall time:** about 1,280 s. This is a planning allowance, not a measurement: 1.5 × the
  archived smoke wall of the matching `rmsnorm` row for each backend and fixture, plus 120 s. It
  excludes container start, image build and queue time.
- **Launch rule:** copy the frozen source to the WSL Linux filesystem (`/tmp/stage7-tier1-20261113T175914Z`),
  check the manifest hash of the copy, and launch detached from there. A paid H200 run needs the user's
  explicit approval.
