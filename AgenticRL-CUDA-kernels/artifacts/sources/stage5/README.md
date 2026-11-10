# Stage 5 frozen source snapshots

Allowlisted public source trees frozen by `scripts/freeze_task_suite.py`
(`--tasks stage5`), each with a `task-source-manifest.json` of SHA-256 per
file. They were taken while Codex finished Stage 5 on 2026-11-09, before any
GPU run.

| Snapshot | Files | Git | Status |
|---|---|---|---|
| `task-stage5-source-20261109T131136Z` | 285 | `fd56edb` | superseded (kept as a record) |
| `task-stage5-source-20261109T131225Z` | 285 | `fd56edb` | superseded (kept as a record) |
| `task-stage5-source-20261109T131446Z` | 285 | `fd56edb` | final pre-GPU freeze; includes the flattened CUDA top-k launch, and matched the committed Stage 5 tree |

The H200 runs used later freezes made at launch time, recorded in each run's
`source-manifest.json` under `artifacts/runs/`.
