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

## RL losses planned for TileGym Tier 2

Four planned tasks from Tier 2 of [`TILEGYM_kernels_task.md`](TILEGYM_kernels_task.md), each computing a loss used in RL or distillation training of a language model. Their operator definitions come from [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) through TileGym's cuTile ports. The contracts were approved on 2026-11-13; **none is implemented yet**. Each task returns a per-row (or per-token) vector and the reduced scalar, both computed in authored kernels, so an error cannot hide inside a mean. All arithmetic is fp32.

Notation: a batch of $N$ rows (tokens), a vocabulary of $V$ entries ($V$ up to 128,256), logits $z_{n,v}$, and log-probabilities $\ell_{n,v} = z_{n,v} - \log\sum_{u} e^{z_{n,u}}$. Every log-sum-exp is computed online over 1,024-wide vocabulary tiles: with running maximum $m$ and running sum $d$, each tile $T$ updates

$$
m' = \max\bigl(m, \max_{v\in T} z_v\bigr), \qquad
d' = d\,e^{m - m'} + \sum_{v\in T} e^{z_v - m'}, \qquad
\log\sum_v e^{z_v} = m + \log d .
$$

### `token_kl_div`: KL divergence between two token distributions

Inputs are log-probabilities: the prediction $y_{n,v}$ and the target $t_{n,v}$. The direction is **KL(target ‖ prediction)**, the convention of `F.kl_div(pred, target, log_target=True)`, and the reduction is batchmean:

$$
\mathrm{KL}_n = \sum_{v=1}^{V} e^{t_{n,v}}\,\bigl(t_{n,v} - y_{n,v}\bigr),
\qquad
\mathcal{L}_{\mathrm{KL}} = \frac{1}{N}\sum_{n=1}^{N} \mathrm{KL}_n .
$$

Outputs: $\mathrm{KL}_n$ per row and $\mathcal{L}_{\mathrm{KL}}$. Swapping the direction fails verification (checked on CPU).

### `generalized_jsd`: generalized Jensen–Shannon divergence

Inputs are log-probabilities of a **student** $Q$ ($\log q_{n,v}$) and a **teacher** $P$ ($\log p_{n,v}$), with a compile-time weight $\beta \in \{0.5, 0.1\}$ and the mixture $m_{n,v} = \beta p_{n,v} + (1-\beta) q_{n,v}$:

$$
\mathrm{JSD}_{\beta,n}
= \beta\,\mathrm{KL}(P_n \,\|\, M_n) + (1-\beta)\,\mathrm{KL}(Q_n \,\|\, M_n)
= \sum_{v=1}^{V}\Bigl[\beta\,p_{n,v}\bigl(\log p_{n,v}-\log m_{n,v}\bigr) + (1-\beta)\,q_{n,v}\bigl(\log q_{n,v}-\log m_{n,v}\bigr)\Bigr],
$$

with $0 \cdot \log 0 = 0$, and $\mathcal{L}_{\mathrm{JSD}} = \frac{1}{N}\sum_n \mathrm{JSD}_{\beta,n}$. TileGym special-cases $\beta = 0$ as KL(P‖Q) and $\beta = 1$ as KL(Q‖P); those are not limits of this formula (at $\beta = 0$ it gives 0) and are not ported. Known limit: swapping student and teacher changes the result by at most 0.9% at $\beta = 0.1$ (measured on CPU), inside the fixed 1e-2 tolerance, so the gate cannot enforce the argument order.

### `fused_cross_entropy`: cross-entropy with ignored positions

Inputs are logits $z_{n,v}$ and target token IDs $y_n \in [0, V)$, or $-100$ for a position to ignore. Label smoothing, class weights, soft-capping and z-loss are off:

$$
\mathrm{CE}_n =
\begin{cases}
\log\displaystyle\sum_{v=1}^{V} e^{z_{n,v}} - z_{n,y_n} & \text{if } y_n \ne -100,\[4pt]
0 & \text{if } y_n = -100,
\end{cases}
\qquad
\mathcal{L}_{\mathrm{CE}} = \frac{\sum_n \mathrm{CE}_n}{\max\bigl(\lvert\{n : y_n \ne -100\}\rvert,\ 1\bigr)} .
$$

This equals `F.cross_entropy(logits, targets, ignore_index=-100)`. Dividing by $N$ instead of the number of kept positions fails verification.

### `grpo_token_loss`: per-token GRPO, DAPO and CISPO policy loss

For sequence $b$ and token position $t$, with the policy's token log-probability $\log\pi_{b,t} = \ell_{b,t,\,o_{b,t}}$ (the log-softmax gathered at the sampled token $o_{b,t}$), the behaviour policy's log-probability $\log\pi^{\mathrm{old}}_{b,t}$, the reference model's $\log\pi^{\mathrm{ref}}_{b,t}$ and a validity mask $\mu_{b,t} \in \{0, 1\}$:

$$
r_{b,t} = \exp\bigl(\log\pi_{b,t} - \log\pi^{\mathrm{old}}_{b,t}\bigr),
\qquad
c_{b,t} = \operatorname{clip}\bigl(r_{b,t},\ 1-\varepsilon_{\mathrm{low}},\ 1+\varepsilon_{\mathrm{high}}\bigr),
$$

with $\varepsilon_{\mathrm{low}}$ = `clip_low` and $\varepsilon_{\mathrm{high}}$ = `clip_high`. The advantage $A_b$ is shared by all tokens of sequence $b$ and is the group-normalised reward of `utils.grpo.group_advantages`, over groups of $G = 8$ rollouts:

$$
A_b = \frac{R_b - \operatorname{mean}_{g}(R_g)}{\operatorname{pstd}_{g}(R_g) + 10^{-6}} .
$$

Per-token loss, by algorithm:

$$
\lambda_{b,t} =
\begin{cases}
-\min\bigl(r_{b,t} A_b,\ c_{b,t} A_b\bigr) & \text{GRPO, DAPO (clipped surrogate)},\[4pt]
-\operatorname{sg}\bigl(c_{b,t}\bigr)\, A_b \,\log\pi_{b,t} & \text{CISPO},
\end{cases}
$$

where $\operatorname{sg}$ is stop-gradient; this is a forward-only task, so its value is just $c_{b,t}$. When `kl_coef` $= \beta_{\mathrm{KL}} > 0$, the k3 estimator of the KL to the reference model is added:

$$
\lambda_{b,t} \mathrel{+}= \beta_{\mathrm{KL}}\Bigl(e^{\,\delta_{b,t}} - \delta_{b,t} - 1\Bigr),
\qquad \delta_{b,t} = \log\pi^{\mathrm{ref}}_{b,t} - \log\pi_{b,t} .
$$

Masked tokens contribute exactly 0 ($\lambda_{b,t} \leftarrow \mu_{b,t}\lambda_{b,t}$). The scalar follows the preset's `loss_agg`:

$$
\mathcal{L}_{\mathrm{seq}} = \frac{1}{B}\sum_{b=1}^{B} \frac{\sum_t \mu_{b,t}\lambda_{b,t}}{\max\bigl(\sum_t \mu_{b,t},\ 1\bigr)},
\qquad
\mathcal{L}_{\mathrm{token}} = \frac{\sum_{b,t} \mu_{b,t}\lambda_{b,t}}{\max\bigl(\sum_{b,t} \mu_{b,t},\ 1\bigr)} .
$$

| Preset (`utils/config.py`) | Clip interval $[1-\varepsilon_{\mathrm{low}},\ 1+\varepsilon_{\mathrm{high}}]$ | Token loss | `kl_coef` | `loss_agg` |
|---|---|---|---|---|
| GRPO | $[0.8,\ 1.2]$ | clipped surrogate | 0.001 | `seq` |
| DAPO | $[0.8,\ 1.28]$ | clipped surrogate | 0 | `token` |
| CISPO | $[0,\ 5]$ | $-\operatorname{sg}(c)\,A\log\pi$ | 0 | `token` |

The CISPO cap is $1 + \varepsilon_{\mathrm{high}} = 5$, not $\varepsilon_{\mathrm{high}}$ itself, which is the convention TileGym gets wrong for this repo (see "TileGym porting conventions" above). Known limit: at the GRPO preset's $\beta_{\mathrm{KL}} = 0.001$ the KL term changes no token beyond the 1e-2 tolerance, so one extra, non-preset case with $\beta_{\mathrm{KL}} = 0.04$ makes it testable (decision Q4).
