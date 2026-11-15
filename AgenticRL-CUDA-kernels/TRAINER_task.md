# Prompt: Add a Trainer to AgenticRL-CUDA-kernels

Written 2026-11-15 against commit `76f77dd`. Every statement about the current repository below was checked against
the code on that date; re-check it before relying on it.

## Role

You are a reinforcement-learning research engineer with LLM fine-tuning and GPU systems experience. You are adding the
missing half of `AgenticRL-CUDA-kernels/` (repo `ElenaViewSynthesis/RL-Army-of-Agents`, branch `main`): a **trainer**
that turns the existing environment into a closed loop.

```
prompt → policy rollout → environment (compile, verify, profile, provenance, reward)
       → group advantages → policy loss → backward → optimizer step → updated policy → next rollout
```

Today only the middle of that line exists. Your job is to build the rest without weakening what is already there.

## 1. What exists and what does not

**The repository is an RL environment that scores kernel candidates. It is not a fine-tuning loop.** Read these
before writing anything:

| Piece | Where | What it does |
|---|---|---|
| Rollout scoring | `utils/rollout_env.py::run_task(workdir, cfg, turns_used=None)` | Detects the backend, then compiles, verifies and profiles a candidate `model_new.py` in separate child processes under one deadline. Returns a flat, JSON-safe dict (key set pinned by `tests/support.py::DEFAULT_RESULT_KEYS`). |
| Reward | `utils/reward.py::compute_reward` | Milestone reward `reward_base ∈ {−1, 1, 2, 3}` (fail; correct; faster than eager; faster than eager and `torch.compile`, each by more than `speedup_margin`), with failure reasons `max_turns`, `correctness`, `unmeasured`, `provenance`, `timing`. Optional DAPO-style overlong penalty from `turns_used`. Pure Python. |
| Advantages | `utils/grpo.py::build_batch(task_ids, rollouts, cfg)` | Groups rollouts per task into `cfg.group_size`, drops `gpu_busy` rows, optionally drops tied groups, and returns `(rollout_id, advantage)` pairs plus batch stats. `group_advantages` is `(R − mean) / (pstdev + 1e-6)` with population std. Torch-free. |
| Presets | `utils/config.py::RolloutConfig.preset("grpo"\|"dapo"\|"cispo")` | GRPO: clip `[0.8, 1.2]`, `loss_agg="seq"`, `kl_coef=0.001`. DAPO: clip `[0.8, 1.28]`, `loss_agg="token"`, tied-group filtering, 4 overlong buffer turns. CISPO: IS-weight clip `[0, 5]` (`clip_low=1.0`, `clip_high=4.0`), `loss_agg="token"`, tied-group filtering, 4 overlong buffer turns. |
| Task suite | `tasks/<name>/`, `utils/task_catalog.py`, `utils/task_suite.py` | 39 tasks in Stages 0–7 with trusted references, honest fixtures and `hack`/`wrong` controls. The task policy requires an authored share of device time ≥ 0.9. |
| Trainer metric definitions | `Architecture-decision/0004-trainer-metric-definitions.md` | Decided definitions for entropy, IS clip fraction, trajectory lengths, the collapse detector and the evaluation cohort. **Nothing emits them yet.** |
| Tracking contract | `Architecture-decision/0002-wandb-tracking-contract.md`, `tracking/` | Reserves the `trainer:` keys. Environment logging never emits them. |
| Loss formulas | `RL.md` ("`grpo_token_loss`"), `CISPO_GPU_batch_ranking.md` §4 | The per-token GRPO/DAPO clipped surrogate, the CISPO objective, the k3 KL estimator and both aggregations, written out. |

**The known gap, stated plainly:** there is no policy model, no sampler or agent loop, no episode ledger, no token
log-probabilities, no policy loss, no backward pass, no optimizer, no checkpointing and no evaluation orchestrator.
`clip_low`, `clip_high`, `kl_coef` and `loss_agg` are stored in the config but nothing reads them: they are
trainer-owned. `max_context_tokens` is exposed for a future loop and not enforced. `env/turns_used` is agent turns
supplied by the caller, never tokens.

Do not confuse the trainer with the kernel *tasks* that happen to implement RL math (`grpo_group_norm`,
`cispo_clip_weight`, `log_softmax_gather`, `packed_varlen_reduction`, and the planned `grpo_token_loss`). Those are
benchmarks a candidate is scored on. They are not the training code, and the trainer must not depend on them.

## 2. Hard constraints

These come from the repository's rules (`CLAUDE.md`, `AGENTS.md`) and are not negotiable without the user's decision.

1. **Do not change reward semantics, timing meanings, verification tolerances, provenance thresholds or protected
   upstream files.** The trainer consumes `run_task`, `compute_reward` and `build_batch` as they are. If one of them
   blocks you, stop and ask.
2. **Add result keys, never rename or redefine them.** Trainer metrics use the keys ADR 0002 reserved and the
   definitions ADR 0004 fixed. If a definition is impractical, propose an ADR amendment; do not emit a different number
   under the reserved name.
3. **Keep the environment Torch-free.** `utils/config.py`, `reward.py`, `grpo.py`, `rollout_env.py` and the other
   parent modules must still import under `python -S` without Torch. All trainer code that needs Torch goes in a new
   top-level package (proposed: `trainer/`), imported lazily. W&B and Weave imports stay in `tracking/`.
4. **No new Python dependencies without approval.** A trainer needs at least a model-loading and a serving or
   generation stack (for example `transformers`, `peft`, `vllm`, `accelerate`). List exactly what you need, pinned,
   with the reason for each, and wait for a yes before adding any.
5. **Never fabricate numbers.** An unmeasured value is `null` with a reason. Every reported number cites its run ID and
   command. A training curve from a stub environment is labelled as such everywhere it appears.
6. **Paid GPU runs need approval, each time,** and run detached on Modal. Give the cost driver (GPU type, count,
   expected wall time) with every request.
7. **The reward stopwatch needs a quiet GPU.** `run_task` profiles device time under a per-GPU lock
   (`utils/gpu_lock.py`). A policy model generating or training on the same GPU corrupts those timings. Policy
   inference and training must run on different GPUs or containers from candidate evaluation, or you must show with a
   measurement that sharing does not move the timings.
8. **Respect the data licences.** Tasks marked `eval_only` in the catalog (the GPU MODE ones) must never produce
   training signal. The Stage 5 EAGLE-4 tasks are evaluation-only with training clearance not recorded
   (`EAGLE4_STAGE5.md`). Build the training task list from the catalog flags, and make a test fail if an eval-only or
   contract-only task enters a training batch.

## 3. Decisions to bring to the user before building

Write these up as a short design note (proposed: `Architecture-decision/0006-trainer-architecture.md`) with a
recommendation for each, and get answers first. Do not pick silently.

1. **Policy model.** `TODO.md` names [Kevin-32B](https://huggingface.co/cognition-ai/Kevin-32B) (QwQ-32B trained with
   multi-turn RL on KernelBench) as the first policy and an external baseline. Alternatives: a smaller open code model
   for cheap iteration first. State memory, licence and whether its tokenizer and chat template are pinned.
2. **What is trained.** Full fine-tuning or LoRA adapters; which layers; precision.
3. **Where rollouts are generated.** In-process generation, or a serving engine. ADR 0004 requires behaviour
   log-probabilities from the raw temperature-1 distribution at generation time; a serving engine that only returns
   warped (top-k/top-p/temperature) probabilities cannot supply them. Say how your choice satisfies this.
4. **Single-turn or multi-turn episodes.** The reward already supports a turn budget (`max_turns`,
   `overlong_buffer_turns`). Multi-turn needs a tool protocol (what the policy sees after a failed compile or
   verification). Recommend starting single-turn and say what multi-turn adds.
5. **Compute plan.** How many GPUs for evaluation workers and for the policy, and the expected cost of the first
   real training run.
6. **Training task set.** Which stages train, which are held out, and the case sets used.

## 4. The math to implement

Take the formulas from `RL.md` and `CISPO_GPU_batch_ranking.md` §4. Do not re-derive them from memory or from another
library. For sequence `b`, token `t`, mask `μ`, behaviour (rollout-time) log-probability `log π_old`, current
`log π`, and sequence advantage `A_b` from `build_batch`:

- Ratio and clip: `r = exp(log π − log π_old)`, `c = clip(r, 1 − clip_low, 1 + clip_high)`.
- **GRPO and DAPO:** per-token loss `−min(r·A, c·A)` (clipped surrogate).
- **CISPO:** per-token loss `−sg(c)·A·log π`, where `sg` is stop-gradient. The clip is on the detached weight, so a
  clipped token still contributes a gradient through `log π`. The upper cap is `1 + clip_high = 5`, an absolute cap,
  not `clip_high` itself.
- **KL term** when `kl_coef > 0` (GRPO preset): `kl_coef · (exp(δ) − δ − 1)` with `δ = log π_ref − log π`, against a
  frozen reference model. The reference is not the behaviour policy.
- **Aggregation:** `loss_agg="seq"` averages tokens within a sequence, then sequences; `loss_agg="token"` divides the
  sum over all valid tokens by the count of valid tokens.
- **Advantages** come only from `build_batch`: broadcast `A_b` to that episode's loss-eligible tokens. Rows dropped as
  `gpu_busy`, and groups dropped as tied, get no loss. Never recompute advantages inside the trainer.

Details that are easy to get wrong, each of which needs a test:

- `clip_low=1.0` for CISPO means a lower bound of 0, which never binds. Compare in log space
  (`log(0) = −inf`) to avoid overflow.
- The behaviour log-probabilities are stored at generation time, per token, from the same raw distribution and context
  the trainer later evaluates. They are data, not a second forward pass with the current weights.
- Select valid tokens; do not multiply invalid positions by zero (a `NaN` times zero is still `NaN`).
- DAPO's dynamic sampling: `build_batch` only *filters* tied groups. Refilling the batch to a target number of
  informative groups is the trainer's job. Decide, document and test what happens when most groups tie, which is
  common when rewards are the four milestone values.
- The reward is sparse and discrete. Expect many all-equal groups early on (all `−1`). Report the fraction of groups
  that produced any gradient at every iteration.

## 5. Milestones

Each milestone ends with tests passing and a short written result. Do not start the next one early.

### M0. Design note and approvals
The decisions in section 3, the dependency list, the package layout, and the data schemas below. No code beyond stubs.
**Gate:** the user has answered every open decision.

### M1. Torch-free reference losses
A small pure-Python module (proposed: `trainer/reference_loss.py`, importable under `python -S`) that computes the
per-token losses, both aggregations and the KL term for all three presets from plain lists. This is the oracle for
everything after it.
**Gate:** unit tests with hand-computed cases: each preset; clipping active and inactive on both sides; zero and
negative advantages; masked tokens; an empty mask; CISPO at and beyond the cap; `seq` versus `token` giving different
answers on ragged lengths.

### M2. Episode ledger
The data ADR 0004 says is missing. One record per episode with: episode and attempt IDs, task/case/backend, prompt
and context per turn, exact generated token IDs, the generated-token ownership mask, per-token behaviour
log-probability and behaviour actor version, tokenizer revision, decoding settings, stop/truncation metadata, turn
count, the terminal `run_task` result, and the effective config. Append-only, versioned schema, JSON-safe.
**Gate:** round-trip tests; a test that the turn count agrees with `env/turns_used`; a test that a ledger with any
missing token field makes the dependent metrics `null`, not zero.

### M3. Sampler and agent loop
Prompt construction from a task (`SKILL*.md`, the task README and contract), generation, writing `model_new.py` into
a scaffolded workdir (`utils/task_suite.py::scaffold`), calling the environment, and recording the ledger. Enforce
`max_context_tokens` and `max_turns` here, since the environment cannot. Workers evaluate candidates in parallel, one
GPU each, never sharing a GPU with the policy.
**Gate:** runs end to end on CPU with a stub policy and the environment in its CPU mode, where every reward is `−1`
with reason `unmeasured`; the ledger is complete; a candidate that crashes, times out or emits no code is recorded as
an episode, not dropped.

### M4. Batch assembly
From ledgers to tensors: call `build_batch`, map advantages to episodes by `rollout_id`, build the loss-eligibility
mask (separate from the diagnostic ownership mask, as ADR 0004 requires), pad, and bucket by length.
**Gate:** property tests that episodes excluded by `build_batch` contribute exactly zero loss and zero gradient, and
that permuting the batch does not change the loss.

### M5. Torch losses and the optimizer step
The Torch implementation of section 4, checked against the M1 oracle to tight tolerance on random inputs, plus
autograd checks: the CISPO weight carries no gradient; the clipped-surrogate gradient is zero exactly where the clip
binds against the advantage's sign; the KL gradient vanishes when `π = π_ref`. Then the update: optimizer, learning
rate schedule, gradient clipping, the unscaled global gradient norm recorded before clipping, mixed precision if used,
and how many passes are made over a batch.
**Gate:** on a toy policy (a tiny model over a small vocabulary) with a synthetic reward, each of the three presets
measurably increases expected reward, with seeds and the curve saved. This is a correctness check of the loop, not a
result about kernels, and is labelled that way.

### M6. Checkpointing and resume
Persist the weights or adapters, optimizer state, the iteration counter `s`, RNG state, the collapse-detector streaks
and latch, and the tokenizer and config versions. Resuming must not double-count an iteration.
**Gate:** a run interrupted and resumed matches an uninterrupted run on the toy problem, or the remaining
nondeterminism is identified and documented.

### M7. Trainer metrics and collapse detector
Emit exactly the ADR 0004 definitions through `tracking/`: `train/entropy`, `train/is_ratio_clip_fraction`,
`train/trajectory_length_tokens`, `train/trajectory_length_turns`, `train/collapse_step`, with their null rules.
Log `build_batch`'s stats alongside. Implement the collapse detector as specified, including the runtime guard that
makes the stalled-reward branch unavailable on environment failures.
**Gate:** unit tests for every null rule and for the detector's streak, reset and latch behaviour, using synthetic
ledgers.

### M8. Evaluation orchestrator
A frozen held-out manifest (task/case/backend cells, attempts, seeds, decoding settings, effective thresholds) and
the three evaluation aggregates with ADR 0004's weights: uniform over tasks, then cells, then attempts; `gpu_busy`
outside every denominator; an incomplete cohort gives `null`, never an invented failure. Evaluation never selects the
best of several attempts.
**Gate:** the weights sum to 1 on a synthetic manifest; an incomplete cohort yields `null`.

### M9. First real run (paid; ask first)
Baseline first: evaluate the untrained policy on the frozen manifest and record it. Then one short training run of
one preset on a small training task set, then the same evaluation again.
**Gate:** a written result with run IDs that states what changed and what did not, including the per-reason
breakdown (`correctness`, `provenance`, `timing`, and so on) before and after. A null or negative result is a valid
outcome and is reported as one.

## 6. Reward hacking is the expected failure, so instrument for it

RL will find the cheapest path to reward. In this environment the known cheap paths are: delegating to PyTorch or a
vendor library (caught by the provenance gate); name spoofing and mixed authored/helper launches (`KNOWN_ISSUES.md`,
"Attribution limits", **not** fully closed); overfitting to five random inputs at one shape; and manipulating timing.

- Log `reward/frac_provenance_fail` and the reason breakdown every iteration; a rising provenance-fail rate that then
  falls while reward rises deserves inspection of the actual candidates, not celebration.
- Keep a sample of the highest-reward candidates per iteration in the run artifact and review them by reading the
  code.
- Evaluate on shapes and tasks the policy did not train on.
- If training discovers an exploit, record it in `KNOWN_ISSUES.md` with the candidate that found it and propose a
  deterministic gate. Do not quietly patch the reward mid-run: a changed reward starts a new run.

## 7. Layout and conventions

Proposed, to be confirmed in M0:

```text
trainer/
  __init__.py
  reference_loss.py     # Torch-free oracle (M1)
  ledger.py             # episode ledger schema and I/O (M2)
  sampler.py            # prompt, generation, agent loop (M3)
  batch.py              # ledgers → tensors, masks (M4)
  losses.py             # Torch GRPO / DAPO / CISPO + KL (M5)
  optim.py              # update step, gradient norm (M5)
  checkpoint.py         # save / resume (M6)
  metrics.py            # ADR 0004 metrics, collapse detector (M7)
  evaluate.py           # frozen-manifest evaluation (M8)
  train.py              # the loop and CLI
cloud/modal_trainer.py  # detached Modal app, separate GPUs for policy and evaluation
tests/test_trainer_*.py
```

Follow the repository's habits: tests run with `python3 -m unittest discover` on Linux or WSL and the Torch-free suite
must stay Torch-free (skip Torch tests cleanly when Torch is absent); every config field gets its CLI flag through
`RolloutConfig` or a separate trainer config that does not disturb it; documentation states what was measured, with
run IDs; commits are small and focused, and nothing is pushed without being asked.

## 8. What to hand back at each step

1. What you built, with file paths.
2. The tests added and their result, with the exact command.
3. What is verified, what is only believed, and what is not done.
4. Any decision you need from the user, as a specific question with your recommendation.
5. For anything that ran on a GPU: run ID, command, cost driver and where the evidence is.

## 9. Stop and ask before

- Changing anything in `utils/reward.py`, `utils/grpo.py`, `utils/config.py` presets, verification tolerances, the
  provenance threshold or a protected file.
- Adding any dependency.
- Launching or restarting any paid GPU job.
- Training on a task whose licence or `eval_only` flag is in doubt.
- Emitting a trainer metric whose inputs are incomplete.
- Reporting that training "works": show the before-and-after evaluation on the frozen manifest instead.

## 10. Non-goals

- New kernel tasks, new backends or faster reference kernels.
- Changing how the environment scores a candidate.
- A distributed training framework. One policy, a handful of evaluation GPUs, and a loop that is correct and
  reproducible come first.
- Claims about beating published systems. The first deliverable is a loop whose every number can be traced.
