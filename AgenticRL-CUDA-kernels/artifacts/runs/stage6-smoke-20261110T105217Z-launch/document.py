import json
from pathlib import Path
project=Path(__file__).resolve().parents[3]
root=project/'artifacts/runs/stage6-smoke-20261110T105217Z'
suite=json.loads((root/'suite/suite.json').read_text())
honest=[r for r in suite['rows'] if r['suite/fixture']=='honest']
bad=[r for r in honest if not r.get('bench/available')]
text='''# H200 Stage 6 smoke review

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
'''
for name in dict.fromkeys(r['suite/task'] for r in honest):
 rows=[r for r in honest if r['suite/task']==name]
 okay=all(r.get('bench/available') and r.get('bench/graph_all_capturable') and
          all(r.get('bench/'+impl+'/graph_replay_ok') for impl in ('eager','compile','generated')) for r in rows)
 text+=f'| {name} | passed, authored 1.0 | '+('passed' if okay else 'failed during replay')+' |\n'
text+='''
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
'''
(root/'REVIEW.md').write_text(text)
p=project/'VALIDATION.md'
s=p.read_text()+'''\n\n## Stage 6 H200 smoke: stage6-smoke-20261110T105217Z (2026-11-10)

The explicitly approved prepared request ran on Modal and completed in
1435.068760015 seconds (remote suite wall time), exit 0. Evidence:
`artifacts/runs/stage6-smoke-20261110T105217Z/REVIEW.md` and raw `suite/suite.json`.
Launcher command and durable function-call ID are under the sibling `-launch/`
directory. Uploaded source is retained with its manifest in the run's `source/`.

Core outcome: **46 passed, 0 failed, 10 unavailable**. All twelve honest CUDA
and Triton cells verified and measured authored provenance **1.0**. The ten
unavailable cells are wrong-skip controls with all-active inputs. Metadata and
SGMV shrink passed benchmark/graph capture/replay on both backends. Eight
other honest benchmark stages failed during timed graph replay with illegal
memory access/device asserts; Modal also logged XID 31/43. Thus this is not
complete graph/performance acceptance. Correctness/provenance counts do not
silently imply successful diagnostics. Failed timings remain unavailable.

Only FP16, rank-8, one-slot, one-slice smoke cases were run. BGMV T=1 and SGMV
T=512 do not establish bf16, mixed ranks, skipped-row or largest-case coverage.
No acceptance/stratified run was launched, and no commit was created. Source
review suggests a possible graph static-input lifetime defect; see REVIEW.md.
This is not a confirmed root-cause or a validated fix.
'''
p.write_text(s)
p=project/'KNOWN_ISSUES.md';s=p.read_text()+'''\n
- **Stage 6 H200 graph benchmarking remains open.** Approved smoke run
  `stage6-smoke-20261110T105217Z` passed all 46 applicable correctness/provenance
  outcomes (10 no-skip controls unavailable); all 12 honest cells have authored
  provenance 1.0. However, eight honest benchmark stages faulted during graph
  replay (illegal memory access/device assertions; Modal XID 31/43). Metadata
  and SGMV shrink benchmarks/replays passed on both backends. The benchmark
  helper does not retain its locally cloned static inputs with the returned
  graph; this is a plausible lifetime issue, not an isolated GPU diagnosis.
  Do not promote this run to full graph/performance acceptance. Frozen source,
  raw failures and review: `artifacts/runs/stage6-smoke-20261110T105217Z/`.
  bf16, mixed-rank/skipped-row and acceptance-grid validation remain pending.
''';p.write_text(s)
p=project/'README.md';s=p.read_text().replace('Implemented; CPU reference/interpreter probes pass, GPU unverified.', 'H200 FP16 smoke correctness/provenance pass; eight graph benchmark stages failed. [Review](artifacts/runs/stage6-smoke-20261110T105217Z/REVIEW.md).')
for line in s.splitlines():
 if line.startswith('| `lora_'):
  name=line.split('`')[1]
  rows=[r for r in honest if r['suite/task']==name]
  if rows:
   okay=all(r.get('bench/available') and r.get('bench/graph_all_capturable') for r in rows)
   replacement='H200 FP16 smoke gates pass; '+('graph replay pass' if okay else 'graph benchmark failed')
   parts=line.split('|');parts[-2]=' '+replacement+' ';s=s.replace(line,'|'.join(parts))
p.write_text(s)
print('Updated smoke review and repository validation documentation')
