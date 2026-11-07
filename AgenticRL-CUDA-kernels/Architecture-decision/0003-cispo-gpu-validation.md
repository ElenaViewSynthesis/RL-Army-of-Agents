# ADR 0003 — CISPO ranking evidence on H200 and the compacted README record

- **Status:** recorded (2026-11-07)
- **Companion to:** [0001-cispo-preset.md](0001-cispo-preset.md) (the decision) — this file records the measured evidence behind it
- **Run:** Modal H200 `20261107T140018Z` — `.venv/bin/modal run cloud/modal_runtime.py --tests`, GPU suite `Ran 31 tests in 503.305s / OK`

## What was measured

`test_every_preset_ranks_the_hack_last` iterates `ALGOS = ("grpo", "dapo", "cispo")`
and passed for the shared CUDA/Triton group and all three DSL groups. The
**measured CISPO group** (honest CUDA, honest Triton, Torch-op hack; rewards
2, 2, −1) ranks the hack last:

| Rollout | Advantage |
|---|---|
| cuda | +0.7071062811869011 |
| triton | +0.7071062811869011 |
| hack | −1.4142125623738022 |

Group stats: `adv/frac_dropped 0.0`, `adv/mean 0.0`, `adv/std 0.9999992928937188`,
`reward/mean 1.0`, `reward/frac_provenance_fail 0.3333333333333333`,
`turns/mean 1.0`, `turns/p95 1.0`. All five honest backends recorded
provenance 1.0 with empty `unmatched_agent_names`.

**What this establishes:** the environment-owned half of CISPO — group-relative
advantages, tied-group handling and reward shaping — behaves identically to
GRPO/DAPO on real hardware, and the provenance gate keeps ranking the hack
last under the CISPO preset. **What it does not establish:** the CISPO loss
itself (`sg(clip(r, 1 − clip_low, 1 + clip_high)) · Â · log π`); there is no
trainer in this repo, and these checks call `build_batch` on measured rewards,
not an optimizer.

## Artifacts and records

- `artifacts/cispo-validation-20261107T140018Z.json` — the structured evidence above
- `artifacts/results-20261107T140018Z.json`, `artifacts/kernel_names-20261107T140018Z.json`,
  `artifacts/environment-20261107T140018Z.json`, `artifacts/gpu_tests.stderr-20261107T140018Z.log`
- `VALIDATION.md` §"CISPO GPU batch-ranking validation: 20261107T140018Z" — the raw summary

## Documentation decision

The README's CISPO paragraph was compacted from ≈1,300 to 852 characters while
keeping every fact and link: the loss formula, "every token keeps a gradient",
the `clip_low=1.0` CISPO-only rule (no lower bound, as in the paper), the
`clip_high=4.0` / IS-weight cap 5 value with the ms-swift → ScaleRL
attribution (MiniMax-M1 publishes no ε_high), and the environment/trainer
split. The decision rationale lives in ADR 0001; the README keeps the
one-paragraph statement and links both sources and this record.
