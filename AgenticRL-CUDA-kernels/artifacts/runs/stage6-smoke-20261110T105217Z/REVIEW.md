# H200 Stage 6 smoke review

Run: `stage6-smoke-20261110T105217Z`; Modal app `ap-ieQ20mtO7rqMdU6rfj9gER`,
function call `fc-01M72WFNZBRTZMQ9EQ6GC30JHH`.

The prepared request was explicitly authorized and executed once. No acceptance
or stratified run was launched. The remote suite completed normally:
**46 passed, 0 failed, 10 unavailable**, with no remote source mismatches.
All twelve honest CUDA/Triton candidates verified and had authored provenance
1.0. Expected Torch substitutions failed provenance; wrong-output and padded-read
controls failed verification. The ten unavailable cells have no skipped tokens,
so the wrong-skip control is not applicable.

This is **incomplete validation**, despite successful core gate counts. Eight of
twelve honest benchmark stages exited with illegal-memory-access/device-assert
errors during CUDA-graph replay. Metadata and SGMV shrink completed benchmarks
and graph replay on both backends. Diagnostics remain separate from rewards;
failed benchmark stages do not erase successful correctness/provenance results,
and are not treated as passing performance evidence.

| Task | CUDA and Triton gates | Benchmark/graph diagnostics |
|---|---|---|
| lora_shrink_bgmv | passed, authored 1.0 | failed during replay |
| lora_expand_bgmv | passed, authored 1.0 | failed during replay |
| lora_segment_metadata | passed, authored 1.0 | passed |
| lora_shrink_sgmv | passed, authored 1.0 | passed |
| lora_expand_sgmv | passed, authored 1.0 | failed during replay |
| lora_fused_sgmv | passed, authored 1.0 | failed during replay |

## Scope and reproduction

Only the frozen FP16 smoke cases were exercised: rank 8, one slot, one slice;
BGMV uses one token and SGMV uses 512. This does not validate bf16, mixed ranks,
skipped-token behavior, largest cases or general memory safety. Do not promote
these results to full acceptance.

`../stage6-smoke-20261110T105217Z-launch/execution-request.json` records the exact
command, frozen source and authorization. `source/task-source-manifest.json`
pins the uploaded source; `runtime.stdout.log` and `nvidia-smi.stdout.log` record
the device/software. Raw case logs and numerical reports are under `suite/`.
Launcher logs preserve Modal GPU XID 31/43 warnings. Failed benchmark timing is
unavailable, not zero-filled. `review.json` records the independent review and
source hash comparison after download.

## Graph failure investigation

The replay failures affect both candidate backends. Source review found that
`utils/benchmark.py::_capture_graph` creates static input clones locally and
returns the graph without retaining those tensor references. Their storage can
be released/reused before later timed replays. This is a plausible lifetime
bug, **not an isolated GPU root-cause finding**: the captured traceback does not
identify which implementation's graph first failed. Retain the raw failing
snapshot; investigate buffer lifetimes and reproduce before claiming a fix.
No scorer, tolerance, kernel, or frozen source was modified during this run.


## Repair audit (2026-11-10; local evidence only)

More precise phase wording: the **benchmark stage crashed during timed rounds**.
All eight saved tracebacks point to frozen `utils/benchmark.py:160`, the call
`_time_calls(graph.replay, args.iters_per_round)`; the raised synchronization
is at line 68. Thus the traces do identify the timed graph-replay call, but
not whether eager, compile or generated was the graph being replayed.
The ATen indexing assert identifies where a bad index was consumed, not what
corrupted it. No standalone repeated CPU forward mutated its inputs in the
new probes. The archived capture helper demonstrably releases static buffers
after return; an owning handle fixes that CPU lifetime repro. Actual GPU
causality and resolution remain unverified until a rerun. Raw suite rows,
timings, downloaded source and historical review.json are unchanged. The
old run used per-element tolerances, not the newly approved Stage 6 gate.
