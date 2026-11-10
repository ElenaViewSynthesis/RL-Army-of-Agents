# Stage 6 local evidence

No GPU execution or measured GPU timing is present in this evidence bundle.
Protocol unit tests can print synthetic timing/provenance fixtures; those are
not measurements. `commands.json` records the original commands and environment.

- `system-suite.log` / `runtime-suite.log`: full discovery runs collected during implementation.
- `final-pure.log`, `final-numerics.log`, `precision.log`: focused post-change checks.
- `offline-compile-final.json`: 28 compiler-only configurations for sm_75/sm_90.
- `rank-probe.json`: 20 synthetic CPU interpreter results and honest/half error distributions.
- `rank_probe.py`: exact script used; its original command/output path is in commands.json.
- `cpu-suite.json`: truthful GPU-unavailable suite records.
- `implementation-files.json`: added/modified implementation paths and final source hashes.
- `source-final/`: definitive frozen public source for review and a separately approved GPU run.
- `source/`: initial preparation snapshot, superseded by source-final before any GPU run; its documentation overstated the number of Stage 6 methods in full discovery. The raw logs were unchanged.
- `h200-smoke-request.json`: prepared request; it has not been executed.

The original relative-error metric and fp16-intermediate expected-failure status
remain diagnostic only. CUDA builds, device execution, graph replay and real
kernel attribution are unverified. No commit was created.
