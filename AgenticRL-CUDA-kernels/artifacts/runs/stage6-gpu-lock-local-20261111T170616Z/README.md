# Stage 6 GPU preparation and GPU contention lock — local evidence

No Modal run, live tracking write or commit was performed. Claude owns
review/commit and launches each detached GPU run after user approval.

The lock serializes profiling, benchmark and diagnostics by device UUID,
with an explicit visibility-key fallback. Shared lock directory defaults
to the system temporary directory/agenticrl-gpu-locks and is overridden
with CUDA_AGENT_GPU_LOCK_DIR. CPU/disabled paths do not acquire. Only
lock wait extends the task deadline; capped wait is gpu_busy, excluded
within original advantage groups. Other reasons/reward inputs are unchanged.
Auxiliary ceiling timing also locks; numerical LoRA checks are verification.

[RUN_PLAN.md](RUN_PLAN.md) lists every case ID, control expectation and
resource assumption. [run-plan.json](run-plan.json) enumerates every row.
The stratified covering subset has six cases per task, 336 rows (312
applicable); acceptance has 56/56 applicable. Estimated wall is 3.91 h
and 2.38 h respectively, with 1200 s per-case caps. Neither is launched.
The widest BGMV acceptance uses T64; the catalog already has T256 at
hidden4096. No catalog shape was added or removed.

The CPU symbolic probe checks 39 largest-case reference/control storage
scenarios. Authored kernels use a reference-storage proxy, not device
execution. Float64 gate/quantile arrays and graph/runtime reserves are
budgeted separately: maximum 106.799 GB, headroom 34.201 GB of 141 GB.
GPU allocator/compiler peaks, timeout margins, new-case correctness
and real contention remain unverified. Full source hashes are recorded
per request after each fresh freeze.

## Validation

system:

```text
Ran 318 tests in 342.187s

OK (skipped=78)
```

runtime:

```text
Ran 323 tests in 1073.095s

OK (skipped=29)
```

The final 9 lock/exclusion tests pass (3.339 s); the wider focused run
passes 71 tests (44.261 s, 6 skips). Exact commands and full logs are
retained. GPU/optional runtime skips are not hardware passes. Protocol
timings and device names in unit-test logs are synthetic. The driver-node
fallback branch was checked separately after full discovery started.
The installed-runtime parent checks under python -S also pass: 12
tests in 2.699 s, OK (`final-runtime-parent.log`).
Initial-target logs retain intermediate missing-tracking-row failures,
and plan-target retains the corrected GB/GiB test-threshold mismatch;
the memory model was not reduced to pass that check. Final runs pass.

[changed-files.json](changed-files.json) and [review.patch](review.patch)
describe the source edits. All seven protected files match upstream.

## Prepared requests and source freezes

- [acceptance-request.json](acceptance-request.json): 277 files, manifest SHA-256 `bcdea24fb705193fd211dc4fe2ed40608daf6e37e62beaf4a156b6a15e9f438c`. Zero frozen/working hash mismatches.
- [stratified-request.json](stratified-request.json): 277 files, manifest SHA-256 `6ab6e18e50b034182cec97552c9c74585fb23e62265a5b1b1087085a27941bd8`. Zero frozen/working hash mismatches.

Both requests remain `launched: false`. Each has a distinct fresh source
freeze, exact detached argv, case IDs, wall estimate and expected outcomes.

Final snapshot checks: both independent freezes contain the same 277 public
source files. The frozen acceptance snapshot passes all 12 parent lock/plan
tests (2.631 s). Manifest copies, exact commands, full logs and a hash
inventory are retained. Nothing remains running from this task.
