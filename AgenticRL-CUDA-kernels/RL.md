# RL notes: trainer metrics

Working notes behind [ADR 0004](Architecture-decision/0004-trainer-metric-definitions.md), which defines the trainer-owned metrics reserved in [ADR 0002](Architecture-decision/0002-wandb-tracking-contract.md).

## CISPO upper clip: ms-swift `epsilon_high=5.0` (ScaleRL)

Reference: [ms-swift CISPO documentation (v4.3)](https://swift.readthedocs.io/en/v4.3/Instruction/GRPO/AdvancedResearch/CISPO.html). Its example sets `epsilon_high=5.0` as an absolute cap on the IS weight, citing ScaleRL for the value. In this repo's `1 + clip_high` convention that is `clip_high=4.0`.

## Approach taken for ADR 0004

ADR number 0004 is available. The current code also has `gpu_busy` and profiler-specific failure reasons, so the definitions need to cover those alongside the listed failures.

- Entropy and IS clipping stay token-denominated across presets.
- Eval rates use all-attempted denominators, except `gpu_busy` rollouts, which are left out of every eval denominator and reported as their own fraction (decided 2026-11-13).
- Geomean speedup is conditional on accepted measurements, with its coverage reported.
- Token masks, policy log-probabilities and gradient data are explicit dependencies.

## Collapse signals

"Collapse" is when RL training goes wrong in a self-reinforcing way. The policy stops exploring and produces nearly the same output every time, or keeps reproducing one trick, and learning stops. The ADR has to define how the trainer detects that, then record the step where it first happened as `train/collapse_step`. Typical warning signs:

- **Entropy drops near zero:** the policy has become almost deterministic.
- **Rewards in a group all become equal:** every attempt at a task scores the same, so GRPO has no difference to learn from. With `filter_tied_groups` on, those groups are dropped, and a rising dropped fraction is a warning sign.
- **Importance ratios drift:** the policy is changing faster than the clipping allows.

`gpu_busy` matters here too. Suppose a group of 8 rollouts has 3 `gpu_busy`. If the collapse signal counts them, it sees 3 identical −1 rewards that have nothing to do with the policy, and could report collapse when the GPU was only busy. Training already drops `gpu_busy` rows before checking for tied groups, so the collapse signals should use the same set of rollouts. Codex should state that explicitly instead of leaving it to whoever writes the trainer.

`profiling` is the second gap. It's a crash in the measurement step after the kernel already passed verification. Codex has to decide whether that counts as the policy's failure, like a compile error, or the environment's, like `timeout`. That one is a judgement call the ADR should make and justify.

## "Stalled rewards" in ADR 0004's collapse detector

ADR 0004 confirms behavioral collapse when, for 20 collection iterations in a row, entropy is at or below 0.1 nats/token **and** one of two things holds: severe importance-ratio drift, or **stalled rewards**. Stalled rewards is the case where the policy keeps producing the same failing kernels, so training has nothing left to learn from.

**What is measured.** Each iteration, take the training groups as `build_batch` formed them (each group is several rollouts of the same task), remove only the `gpu_busy` rows, and drop groups left empty. Before any tied-group filtering, compute three numbers:

| Signal | Meaning | Stalled when |
|---|---|---|
| `V_s` | Average reward variance within a group, each group weighted equally | `≤ 1e-12`: rewards inside a group are effectively identical |
| `Q_s` | Fraction of groups whose rewards are exactly equal | `≥ 0.95`: almost every group is tied |
| `S_s` | Fraction of rollouts that were accepted (passed with valid timings) | `≤ 0.10`: almost nothing succeeds |

All three must hold at once.

**Why each condition is needed.**

- **Tied groups give GRPO nothing to learn.** Advantages are each reward minus its group's mean, so a group whose rollouts all score the same gives every rollout an advantage of zero. `V_s` and `Q_s` measure how much of the batch is in that state. The `1e-12` cutoff is the square of the `1e-6` stabiliser in `group_advantages`, so it means "zero within the advantage code's own precision".
- **Tied alone is not collapse.** A policy that solves every task scores the same on every rollout too, and its groups tie as well. That is success. The `S_s ≤ 0.10` condition separates the bad case, where everything ties because everything fails, from the good one, where everything ties because everything passes.
- **Low entropy is required as well.** Flat, failing rewards on their own could just mean the tasks are too hard at the start of training. Collapse also needs the policy to have stopped exploring.
- **Persistence: 20 iterations in a row.** A single bad batch does not count. Any iteration that misses a condition, or is missing data, resets the count.

**When the check is switched off.** The stalled-rewards check only counts if every rollout in the batch failed for a reason that is the policy's fault, or succeeded:

- **Allowed:** the three success reasons, plus `compile`, `verification`, `correctness`, `provenance` and `max_turns`.
- **Not allowed:** `gpu_busy`, `timeout`, `unmeasured`, `profiling`, `timing`, `diagnostic_profile`, `workdir`, `backend` or an unknown reason anywhere in the batch makes the check unavailable for that iteration. These are environment or measurement problems, not evidence about the policy.
- **Runtime check:** the iteration's runtime record must also show a working GPU, driver and toolchain for every backend used. A broken `nvcc` makes every rollout fail with `compile`, which would otherwise look exactly like a policy that writes bad code.

An unavailable iteration resets the 20-iteration count, the same as a failed one.

**Example.** One iteration has 16 groups of 8 rollouts. 15 groups are all `compile` failures (reward −1 each), and one group has two `correct` rollouts and six `compile` failures. Then:

- `Q_s` = 15/16 = 0.94, just under the 0.95 cutoff, so the iteration is not stalled.
- `S_s` = 2/128 = 0.016.

If the last group also collapses to all `compile`, `Q_s` becomes 1.0 and `V_s` becomes 0, so the iteration counts as stalled, provided entropy is also at or below 0.1. Twenty such iterations in a row confirm collapse, and `train/collapse_step` records the twentieth iteration's index, not the first.

**Not part of the detector.** `adv/frac_dropped` (the fraction of groups removed by `filter_tied_groups`) is reported only as context. GRPO doesn't drop tied groups, so that number would mean different things under different presets. `Q_s` is computed the same way for every preset instead.

## TileGym porting conventions (approved 2026-11-13)

Four conventions found while surveying NVIDIA TileGym v1.5.1 for cuTile ports. Codex's survey note, kept verbatim: "RoPE mutates Q/K in place, GEGLU gates the second half of its input, and TileGym's JSD uses special KL behavior at beta 0 and 1. Its CISPO code also treats the upper clip as a direct cap, while this repo uses 1 + clip_high. I'll make each convention explicit in the proposal." Each is now pinned in a task contract (`HANDOFF_CODEX_TILEGYM.md`, decisions Q1-Q10 approved).

| Convention | What TileGym does | What this repo does | Why it matters for RL |
|---|---|---|---|
| **RoPE in place** (`ops/cutile/rope.py:36-45, 126-134`) | Rotates Q and K in place and returns them. | `rotary_embedding` returns **new** `q_rot`, `k_rot`; inputs are read-only. The verifier compares outputs only, so in-place mutation is caught by source tests (Q5 adds an input-unchanged check for new tasks). | A kernel that overwrites its inputs still passes an output-only check; the gate must not reward that. |
| **GEGLU half** (`ops/cutile/activation/geglu.py:82-83, 105-112`) | GELU on the **second** half, the first half is the multiplier (the opposite of SiLU-and-mul, which gates the first half). TileGym's `approximate="none"` still uses the tanh formula: cuTile 1.8.0 has no `erf`. | `geglu_tanh` is tanh-only and says so; tanh and erf GELU differ by at most 4.7e-4, inside the tolerance, so this task cannot serve as evidence about the exact-erf `layernorm_residual_gelu` contract. | A split or formula mismatch would silently pass or silently fail depending on the random inputs. |
| **JSD at beta 0 and 1** (`suites/liger/cutile/jsd.py:57-58, 88-99`) | beta = 0 returns forward KL(P‖Q), beta = 1 returns reverse KL(Q‖P); these branches are not limits of the general formula (which gives 0 at beta = 0). | `generalized_jsd` uses beta in {0.5, 0.1} as a compile-time constant and does not port the special cases; the KL direction lives in `token_kl_div` (KL(target‖pred), batchmean). The student/teacher order cannot be verified at the fixed tolerance (Q8: accepted and documented). | Distillation and policy-divergence losses are direction-sensitive; the contract has to say which argument is the student. |
| **CISPO cap** (`suites/liger/cutile/grpo_loss.py:146` vs `159`) | GRPO clips at `1 + eps_high`, but the CISPO branch caps at `eps_high` **directly**, with no lower bound. | The CISPO preset stores `clip_high = 4.0` meaning a cap of **5** (`1 + clip_high`, ADR 0001, ms-swift `epsilon_high = 5.0`). `grpo_token_loss` takes explicit `low = 1 - clip_low`, `high = 1 + clip_high` for every algorithm; negative controls `_wrong_cap4` and `_wrong_cap6` catch both naive ports (they change 6-8 % of tokens and 7-9 % of the scalar, measured on CPU). | An off-by-one in the IS-weight cap changes the gradient scale of every high-ratio token; it is exactly the convention mismatch ADR 0001 recorded. |

Two further decisions from the same review: the loss tasks return `(per_row, scalar)` so errors cannot hide in a mean and the reduction runs in an authored kernel (Q3); and the GRPO KL term at the preset `kl_coef = 0.001` is below the tolerance, so one non-preset case with `kl_coef = 0.04` makes it testable (Q4; the value is DeepSeekMath's beta to our recollection and must be checked before it enters the catalog).
