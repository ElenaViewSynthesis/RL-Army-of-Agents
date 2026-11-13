# cuTile local implementation and prepared H200 requests

Nine source candidates; no cuTile compilation or device execution. See root
VALIDATION.md for scope and raw suite summaries. All timing/speedup/provenance
values for these candidates remain null with reason `No cuTile task GPU run launched`.

Both final suites: system 339 tests / 565.597 s / OK (78 skipped); runtime
344 tests / 1444.943 s / OK (29 skipped). Full logs and intermediate failures
are retained. `local-review.json` records protected hashes, unchanged catalog/
default backends, historical README preservation and changed-file hashes.
`api-review.json` + `wheel-source.json` record pinned API inspection only.
`source-review.log` compares modified files with the session baseline.
`source-checks.json` and `source/*/task-source-manifest.json` pin the freezes.
`evidence-scan.json` records the local pattern scan; review before committing.

| Request | Case set | Rows | Estimated wall | Exact request |
|---|---|---:|---:|---|
| tier1 | smoke | 15 | 14.1 min | [tier1-request.json](tier1-request.json) |
| tier2 | smoke | 6 | 6.5 min | [tier2-request.json](tier2-request.json) |
| tier3 | smoke | 6 | 10.7 min | [tier3-request.json](tier3-request.json) |
| gemm-acceptance | acceptance | 3 | 10.4 min | [gemm-acceptance-request.json](gemm-acceptance-request.json) |

Estimates and per-row basis are in `run-plan.json`; they use historical Triton
walls plus compile/startup allowances, not cuTile timings. Requests are prepared,
not launched. Claude reviews and obtains approval, launches one detached H200
request, reports every outcome and actual device symbol, then stops. Do not
launch next tiers, retry, tune or promote automatically. No tracking write or
commit occurred. Default rows, catalog shapes/dtypes and all reward/provenance/
verification contracts remain unchanged. The flagship request additionally
requires the unchanged paired cuBLAS floor (>=0.6) and authored share 1.0.

Frozen source code is outside this public bundle under `artifacts/sources/cutile/`.
Only manifests are retained here. Root `HANDOFF_CODEX_CUTILE.md` is an ignored
working note for Claude; it lists all files, launches, choices and open questions.
