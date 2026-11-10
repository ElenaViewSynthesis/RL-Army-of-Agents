# Stage 5: documented speculative-tree workloads

These evaluation adaptations were authored with Codex on 2026-11-09 from the
[EAGLE-4 kernel documentation](https://github.com/eagle-hq/EAGLE-4/blob/14fb79a5a4562ebe671444d7841253b889301040/docs/kernels.md).
No source under `eagle_harness/` was fetched, inspected, copied or ported.
The upstream revision, downloaded documentation/legal files, retrieval time and
SHA-256 digests are in `legal/eagle4/source.json`.

## Source, license and attribution

This project includes or is derived from Eagle 4 technology developed by Eagle AI
Corp. It is not an official Eagle AI Corp project. References identify the origin
of these workloads, not affiliation, certification or endorsement.

The **Eagle 4 Source-Available License** applies to these documentation-derived
adaptations. It is not an OSI-approved open-source license. Section 2 expressly
permits evaluation, benchmarking and derivative works, subject to its conditions;
it does not prohibit the requested noncommercial evaluation workloads. Sections
3 and 4 retain competing-offering/commercial restrictions and distribution,
source, attribution and modification-notice obligations; section 8 requires
compliance records. Evaluation-only does not waive those conditions or authorize
commercial or competing use. Training use is deliberately not cleared here.

Preserved upstream files: [LICENSE](legal/eagle4/LICENSE),
[NOTICE](legal/eagle4/NOTICE), [Trademark Policy](legal/eagle4/TRADEMARKS.md),
[third-party notices](legal/eagle4/THIRD_PARTY_NOTICES.md), and `LICENSES/` beneath
that directory. Downloading, cloning, forking, copying, accessing, modifying,
using, hosting, distributing, or otherwise exercising licensed rights constitutes
acceptance of that license and incorporated trademark policy. The third-party
Apache/BSD/MIT texts do not relicense the Eagle documentation or the whole project.
Public source freezes preserve these materials. This is an independent evaluation
adaptation within AgenticRL CUDA Kernels, not an official Eagle distribution.

Modification/provenance notice: the AgenticRL CUDA Kernels repository maintainers,
using Codex, authored the new `tasks/eagle_*/` files, `utils/eagle_contracts.py` and
`utils/eagle_reference.py` on 2026-11-09. They implement the explicit contracts below from documented behavior,
with expanded grids, sampling, simplified channel layouts and functional state
outputs. Existing task catalog, input dispatcher, suite, README renderer and
freeze script gained registration, alternate-fixture handling, documentation and
notice preservation. The repository's existing stock scorer is unchanged.

## Differences from the paraphrased brief

The actual page names fp32 **products** for candidate re-scoring, not a promise of
real-number exact sums. It does not fully specify candidate shortlist size,
frontier tie ordering, tensor layouts, recurrence equations or sampling. The
contracts below specify those details locally; they are not claims of upstream
ABI equivalence or bitwise equivalence to unexecuted engine code.

The documented greedy accept-tail runs batch-size-1 rounds on SGLang. B=8/32
and rejection sampling are repository extensions. Mamba2 uses the same tree
conv/ancestor tables; its selective state update remains out of scope. The GDN
walk and fp32 first-child-chain replay are documented alternatives; Gluon is out
of scope. The original page also describes tree-step cost-model work deltas,
L2 prefetch, native candidate-tail code, and engine integration; these contracts
cover only the explicitly requested engine-free subset, not those facilities.
Exact integer verification was already approved on 2026-11-09. Floating tolerance
remains atol=rtol=1e-2, including recurrence state snapshots. No bitwise fp32-state
claim or new RolloutConfig field is introduced.

## Common conventions and case grids

All tensor inputs are contiguous in their declared layouts. All trees are ordered: `parents[B,N]` int32, root parent -1, and every other
parent is smaller than its child. Shared committed request state precedes the
root. The generator supplies chain, wide, balanced and ragged shapes at
N=1/4/8/16/32/64 and B=1/8/32. Kernels read the supplied parent tables; they do
not regenerate trees or infer parentage from a shape label. Siblings cannot see
each other's states or KV. Ancestor tables are self-first, then parent toward
root, with -1 padding. Constructors/state_dict match, and inputs are not mutated.

Smoke cases are first. `--case-set acceptance` picks the final stage-5 grid case:
B=32/N=64/ragged for tree tasks, with the largest remaining dimensions. These
can be large; they are not automatically launched. Vocabulary grids are
32000/128256/151936; candidate k=8/16; attention has 8 KV heads, GQA ratios
1/4/8, head dimensions 64/128, contexts 512/4096/32768. Row windows are
min(context,4096); verify attends the whole committed context.

All nine tasks have `eval_only=True`; `--purpose train` refuses them. The
held-out-evaluation restriction includes avoiding RL feedback disguised as eval.

## Exact operator contracts

- **eagle_candidate_topk:** hidden[B,H], weight[V,H], both bf16, H=4096.
  Compute fp32-product vocabulary scores. Select k by decreasing score, breaking
  equal scores by lower token ID; return int64 IDs[B,k]. Re-score those selected
  weight rows with fp32 multiply/reduce; return fp32 scores[B,k] in selection
  order. The Triton solution merges shard shortlists; CUDA uses a conservative
  whole-vocabulary selector. Neither calls vendor GEMM. Close but unequal scores
  are not ties; floating sums are verified at the existing tolerance, IDs exactly.
- **eagle_tree_step:** fp32 scores[B,N], children[B,N,k], int32 parents[B,N].
  Select N of N*k sums, with flat `(parent,child)` index breaking ties. Return
  fp32 frontier scores[B,N], int32 selected parent indices[B,N], and
  paths[B,N,N+1], plus int32 selected flat child indices[B,N]. A selected new node has ID N+rank; its path starts with that ID
  and continues with its selected parent's self-first ancestor path. Cost-model
  work deltas are intentionally outside this operator.
- **eagle_tree_row_attention / eagle_tree_verify_attention:** bf16
  Q[B,Hq,N,D], committed K/V[B,Hkv,S,D], tree K/V[B,Hkv,N,D], parents[B,N].
  Query heads are grouped consecutively onto KV heads. Each query sees the last
  window committed keys and its own self-inclusive ancestor KV; fp32 scaled
  logits/softmax/PV, bf16 output[B,Hq,N,D]. Also return exact paths[B,N,N].
  Split partial maxima/normalizers/weighted sums are merged stably. Row attention
  uses a fixed 4096 cap, not a dynamic engine cost-model split policy.
- **eagle_gdn_tree_conv:** bf16 X[B,N,C], request history[B,3,C] (oldest first),
  fp32 weights[C,4] and bias[C], parents[B,N]. The current token multiplies
  weight[:,3], its predecessor weight[:,2], etc. Before the root, windows use
  history[-1], history[-2], history[-3]. No activation is fused. Return fp32
  output[B,N,C], the int32 parent table, and windows[B,N,4]. Negative window
  entries -1/-2/-3 denote those request-history tokens, not missing values.
- **eagle_mamba2_tree_conv:** the same convolution with X/output[B,C,N]
  contiguous, and identical history/weights/parent/window contracts. C=192/384
  represents concatenated x/B/C channel segments of width 64/128, one logical
  group each; this simplified channel layout is not a full NemotronH layer or
  selective-state-update implementation. GDN uses token-major channels, Mamba2
  channel-major. Both use the same parent/window derivation.
- **eagle_gdn_tree_recurrence:** fp32 Q/K[B,N,H,Dk], V[B,N,H,Dv],
  gates[B,N,H,2]=(alpha,beta), initial[B,H,Dk,Dv], parents[B,N]. No gate sigmoid
  is hidden: gates are already transformed. With parent state S, compute
  `T=alpha*S`, `r=V-K^T*T`, `S'=T+K outer (beta*r)`, `out=Q^T*S'`.
  Return fp32 outputs[B,N,H,Dv], snapshots[B,N,H,Dk,Dv], int32 parents and paths.
  Inputs use bounded gates and scaled Q/K; no state is modified in place.
  The default Triton path walks nodes over 16-wide value tiles. The alternate
  `chain_replay` path builds first-child and leaf-owner tables, replays each
  leaf path from request state, and stores only nodes owned by that first-child
  chain. Shared prefixes are recomputed; each node has exactly one writer.
  CUDA provides the node walk. H=1/4, Dk=Dv=64/128. Snapshot tolerance is the
  existing 1e-2, not exact equality; no new tolerance was approved or added.
- **eagle_accept_tail:** int32 tokens[B,N], parents[B,N], target argmax[B,N+1],
  slots[B,N]. Target row 0 belongs to the virtual root; row j+1 belongs to node j.
  Start at virtual parent -1; take the lowest-index child matching the current
  target. Stop at a mismatch or leaf. Return int32 lengths[B], paths[B,N],
  bonus[B], compacted slots[B,N], and parents[B,N]. Paths/slots are -1 padded;
  bonus is the stopping node's target argmax. Large slot IDs exercise exact
  comparisons; no actual KV array is modified by this functional operator.
- **eagle_accept_tail_sampled:** tokens/parents/slots as above; normalized fp32
  target/draft[B,N+1,V], acceptance uniforms[B,N], bonus uniforms[B,N+1].
  Consider siblings in node order. For each proposal token t, accept if
  `u < min(1,residual[t]/draft[t])`; a zero proposal mass has acceptance ratio 0.
  After rejection, use normalized `max(residual-draft,0)` for the next proposal.
  First accepted child wins; continue from that child's unmodified target row.
  If none accepts, draw the bonus from the corrected residual by the first CDF
  strictly greater than its uniform (last token fallback for rounding).
  Return the five greedy-shaped integer outputs plus fp32 bonus probability[B]
  under the stopping residual. For fixed work accounting, every edge's candidate
  correction is computed, but only active rejections commit updates. Every
  parent's bonus CDF is evaluated, then the winning path is gathered. Corrected
  mass reductions and CDF sums use fp64, probabilities remain fp32. Proposal
  probabilities must match the distribution that generated the draft tokens;
  generated inputs enforce that. Uniforms come from a local seed-17 generator;
  no candidate-side RNG or new configuration field is needed. Fixed inputs and
  uniforms give deterministic outputs. Zero-mass correction uses denominator 1;
  it is unreachable on a valid positive-mass rejection with u in [0,1).

## Work accounting and controls

`workload()` states each task's FLOP and logical-byte convention. A multiply-add
counts as two FLOPs, comparisons/index traversal as zero, transcendental softmax
work is excluded from attention FLOPs. Greedy acceptance has zero floating work.
Sampling counts all-edge residual subtraction/reduction/division plus all-parent
CDF sums, independent of the winning path. Recurrence rates count the node-walk
mathematics, not duplicated chain-prefix work. These are algorithmic rates, not
executed instruction counts. Bytes count each supplied input and padded output
once, including integer metadata; they do not measure cache/HBM traffic or scratch.
Unit tests compare byte counts against actual reference tensor sizes.

Each task has `honest`, `hack` and `wrong` fixtures. Every tree task also has
`wrong_ancestor`: it deliberately shifts input parent links down by one (clamped at -1 for
finite traversal) and offsets returned parent/ancestor metadata by one. The
metadata corruption also makes the one-node smoke control observable. Verification
must reject it before profiling; this does not claim every parent-link bug is
detected. Independent reference/branching tests
also check that actual ancestor links determine outputs. The Torch-op hack must
verify then fail authored-only provenance. Names remain heuristic attribution.

Use `--fixtures all` to include applicable wrong-ancestor controls and the GDN
Triton `chain_replay` fixture. Explicit `--fixtures chain_replay --backends triton`
is supported for recurrence only. CPU runs establish correctness, not GPU passes.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover
# On an approved GPU runtime, after freezing public source:
python -m scripts.validate --suite stage5 --device cuda --purpose eval \
  --case-set smoke --fixtures all --timeout-s 600
```

No H200 run is authorized or launched by the implementation task. CUDA builds,
real device provenance, GPU numeric behavior, and the larger case grids remain
unverified until separate GPU evidence is recorded.
