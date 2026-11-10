# Run evidence

Every GPU run and local test log from 2026-11-09 on lives here, one folder per
run, named by its run ID (`task-<scope>-<UTC timestamp>`).

- **Modal GPU runs:** `summary.json`, `request.json`, `source-manifest.json`
  (the frozen source the run used), `suite/suite.json`,
  `suite/progress.jsonl`, the per-case `suite-case.json` files, benchmark
  samples, and the stage logs. Scaffolded sources, caches, binaries and
  profiler sidecars stay on the Modal volume (`agenticrl-cuda-artifacts`).
- **Local test runs:** `system.log` (system Python) and `venv.log` (venv).

Runs before 2026-11-09 remain at their original paths under `artifacts/`.
