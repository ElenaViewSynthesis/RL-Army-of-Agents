from pathlib import Path

def edit(path,fn):
 p=Path(path);raw=p.read_bytes();s=raw.decode().replace('\r\n','\n');s=fn(s)
 p.write_bytes(s.replace('\n','\r\n').encode() if b'\r\n' in raw else s.encode())
rule='''## Tolerances and gates

- **Approved Stage 6 arithmetic verification:** for every floating output,
  `max(abs(actual-ref)) <= c * max(abs(ref))`, where activation dtype selects
  c=0.005 (fp16) or c=0.01 (bf16), including fp32 shrink buffers. This replaces
  the old per-element atol/rtol check for Stage 6 only.
- Skipped rows of y (`idx=-1`) must be bit-identical to y_in and are excluded
  from both reductions. Actual NaN/inf where reference is finite fails.
  A zero checked reference requires exact equality; an all-skipped batch
  still checks every skipped y byte.
- Metadata and all integer outputs stay exact. Stock and Stages 0–5 retain
  their existing comparison paths. The suite forces `relative_error_bounds`
  only for Stage 6 arithmetic; it defaults off and has a matching CLI flag.
- Case JSON includes per-output bound, checked reference maximum, errors and
  decision from the verification subprocess. Its exit status gates rewards;
  candidate-writable report files never feed a reward.
- The fp16-intermediate fixture remains diagnostic only; record whether it
  would violate the approved bound, without an expected-failure gate.
- The authored-only provenance gate applies: no cuBLAS/cuDNN time in honest
  solutions.

'''
edit('MULTI_LORA_kernels_task.md',lambda s:s[:s.index('## Tolerances and gates')]+rule+s[s.index('## Diagnostics',s.index('## Tolerances and gates')):])
edit('MULTI_LORA_STAGE6.md',lambda s:s.replace('intermediate errors appear side by side. No tolerance or reward is changed.', '''intermediate errors appear side by side. The approved relative rule below gates
Stage 6 arithmetic verification; the fp16-intermediate comparison stays diagnostic.''')+'''

## Approved verification and replay repair (2026-11-10)

`RolloutConfig.relative_error_bounds` defaults off. The progressive suite enables
it only for Stage 6 arithmetic and selects `utils.lora_verification` in the same
verification subprocess stage. For each float output the maximum absolute error
must be at most 0.005 times the checked reference maximum (fp16 activations), or
0.01 (bf16 activations), even when shrink returns fp32. Skipped y rows are checked
bit-for-bit against y_in and excluded from reductions; zero reference requires
exact equality. Non-finite errors fail; integers retain exact verification.
This replaces the prior per-element floating check, without changing the stock
or Stage 0–5 paths, speed reward or provenance semantics. Detailed gate decisions
are recorded in `.cache/relative-error-bounds.json` and `suite-case.json`.
Diagnostics include the half intermediate's hypothetical decision but do not
turn it into a failure fixture or acceptance count.

The benchmark capture helper now returns an owning `CapturedGraph` handle that
keeps static input clones and captured outputs alive for all timed replays.
A CPU weak-reference repro fails on the archived helper and passes on the fix.
Repeated CPU reference/Triton calls preserve every input byte for 40 iterations;
no caller mutation was found in these probes. Native CUDA paths and the actual
H200 crash resolution require another approved GPU run. Torch graph-pool output
allocations remain supported; this fix preserves buffers and does not change
forward signatures or introduce in-place y updates. See the PyTorch 2.14
[graph lifetime and allocator contract](https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management).
''')
edit('README.md',lambda s:s+'''

Stage 6 correction pending GPU review: the approved activation-dependent global
relative bound now gates arithmetic verification (fp16 0.005, bf16 0.01). Skipped
y rows are exact and excluded from the bound. Stock/Stages 0–5 are unchanged.
The replay helper retains static buffers for the capture lifetime. Local
regression evidence and remaining GPU limits are recorded in VALIDATION.md.
The earlier H200 smoke used the old floating check; it does not validate this
new gate or replay repair. No rerun is launched as part of the repair.
''')
# Correct the derived review, leaving raw smoke results/snapshots untouched.
edit('artifacts/runs/stage6-smoke-20261110T105217Z/REVIEW.md',lambda s:s+'''

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
''')
