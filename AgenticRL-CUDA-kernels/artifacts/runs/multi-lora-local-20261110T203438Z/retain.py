import hashlib,json,shutil
from pathlib import Path
root=Path(Path('/tmp/stage6-repair/bundle-path').read_text())
for p in Path('/tmp/stage6-repair').iterdir():
    if p.is_file() and p.suffix in ('.json','.log','.py') and p.name not in ('system-suite.log','runtime-suite.log'):
        shutil.copyfile(p,root/p.name)
(root/'README.md').write_text('''# Stage 6 gate and benchmark repair — local evidence only

No Modal run or commit was performed. Any timing/provenance values printed by
protocol unit tests are synthetic fixtures, not GPU measurements.

## Confirmed local findings

- The archived capture helper does not retain static tensors after return.
  `tests/test_graph_lifetime.py` reproduces released buffers after GC using a
  weak-reference graph. The owning `CapturedGraph` fix survives 40 rotating
  replays and releases its references when the handle is destroyed. This is
  a confirmed CPU ownership defect; actual H200 causality/resolution is still
  unverified. PyTorch's 2.14 CUDA notes require long-lived capture input/output
  references and support graph-private allocator pools.
- Repeated synthetic CPU tests performed 2,640 calls (66 scenarios × 40).
  Every input byte and every output remained stable. Stage 6 reference/Triton
  probes cover both dtypes, mixed ranks and skipped/all-skipped patterns;
  cheap Stage 5 reference/Triton probes cover all nine tasks. Mutation was not
  reproduced. Native CUDA and CUDA-compiled references remain untested locally.
- `failure-phase.json` audits all eight old failures against the frozen source:
  the benchmark stage crashed in timed rounds at the graph.replay timing call,
  not an identified implementation's graph. Null graph fields reflected a
  stage crash before serialization; they do not establish a pre-capture failure.

## Approved gate

Stage 6 arithmetic uses max error ≤ c × checked max reference, c=.005 for
fp16 activations and .01 for bf16, including fp32 shrink buffers. Skipped y
rows are byte-exact and excluded from the bound. Zero references are exact;
non-finite errors fail. Integers stay exact. The gate replaces the old float
check; stock and Stage 0–5 paths are unchanged. All six ADR config row types
and CLI/config validation are wired. Gate decisions are saved in case JSON,
but candidate-writable reports do not compute a reward; subprocess exit status
does. fp16-intermediate remains diagnostic with a hypothetical gate decision.

32 supplemental interpreter probes at ranks 64/256 and mixed ranks all pass
the approved bound; maximum observed fraction of the bound is 0.5649717514.
No honest tested case required relaxation. These are small CPU probes, not
catalog GPU acceptance. Full-grid numerical risk and CUDA memory behavior are
still unverified. `verifier-integration.json` records actual verifier exit 0
for honest and exit 1 for wrong_padding/wrong_skip, with the gate reports.

## Files

`commands.json` records exact commands. `changed-files.json` lists changes.
Raw full-suite logs, focused logs, mutation traces, gate reports and phase
proof are retained here. `source/` will contain the definitive public freeze.
`h200-smoke-request.json` is prepared for Claude review/launch only, after user
approval. No task forward ABI or in-place y semantics changed. The approved
Stage 6 numerical verification criterion intentionally changed.
''')
Path('/tmp/stage6-repair/bundle-path').write_text(str(root))
print(root)
