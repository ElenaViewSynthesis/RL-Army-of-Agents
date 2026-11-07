# CISPO GPU-batch ranking in this codebase

*Explainer note (2026-11-07). Companion to [Architecture-decision/0001-cispo-preset.md](Architecture-decision/0001-cispo-preset.md) (the decision) and [Architecture-decision/0003-cispo-gpu-validation.md](Architecture-decision/0003-cispo-gpu-validation.md) (the measured evidence).*

"CISPO GPU-batch ranking" is the same group-relative advantage machinery the
environment already uses for GRPO/DAPO — CISPO changes nothing in how
advantages are computed. What the codebase validates is that, under the CISPO
preset, the reward-hacking rollout still ranks last in a group **measured on
real hardware**. The pipeline, end to end:

## 1. Each rollout in the group earns a discrete reward

Every member of the group is a full `run_task` evaluation on the GPU
(compile → verify → profile). The reward comes from `compute_reward`
(`utils/reward.py`):

$$
R(\tau) \;=\;
\begin{cases}
-1, & \text{correctness / provenance / timeout / turn-cap failure},\\[2pt]
1, & \text{correct, not faster than eager},\\
2, & \text{faster than eager by more than the margin},\\
3, & \text{faster than both eager and } \texttt{torch.compile}.
\end{cases}
$$

In the measured H200 group: honest CUDA → $R=2$, honest Triton → $R=2$,
TorchOp reward hack → $R=-1$ (provenance $0.0$ against the $0.90$ gate).

## 2. `build_batch` groups the rollouts and standardizes rewards within each group

`utils/grpo.py` is Torch-free and algorithm-agnostic — nothing in it knows
what CISPO is. It:

- **buckets rollouts by `task_id`** and splits them into whole groups of
  $G = $ `cfg.group_size` (incomplete groups are errors). For a group
  $\mathcal{G} = \{\tau_1, \dots, \tau_G\}$ with rewards
  $\{R_1, \dots, R_G\}$:
- **applies the tied-group filter** `filter_groups(groups, cfg.filter_tied_groups)`
  — the one place the CISPO preset changes environment behavior. CISPO and
  DAPO set `filter_tied_groups=True`, so the group is **dropped** whenever

$$
\min_{i \in \mathcal{G}} R_i \;=\; \max_{i \in \mathcal{G}} R_i
\qquad (\sigma_{\mathcal{G}} = 0 \;\Rightarrow\; \text{no learning signal}),
$$

  while GRPO keeps it with all-zero advantages.
- **computes group-relative advantages** with `adv_norm="std"`:

$$
\hat{A}_i \;=\; \frac{R_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}} + \varepsilon},
\qquad
\mu_{\mathcal{G}} = \frac{1}{G}\sum_{i=1}^{G} R_i,
\qquad
\sigma_{\mathcal{G}} = \sqrt{\frac{1}{G}\sum_{i=1}^{G}\bigl(R_i - \mu_{\mathcal{G}}\bigr)^2},
\qquad
\varepsilon = 10^{-6},
$$

  using the **population** standard deviation, so advantages are
  zero-centered within the group: $\sum_{i} \hat{A}_i = 0$.

For the measured rewards $(2,\,2,\,-1)$: $\mu_{\mathcal{G}} = 1$,
$\sigma_{\mathcal{G}} = \sqrt{2} \approx 1.4142$, giving

$$
\bigl(\hat{A}_{\text{cuda}},\, \hat{A}_{\text{triton}},\, \hat{A}_{\text{hack}}\bigr)
= \left(\tfrac{1}{\sqrt{2}},\, \tfrac{1}{\sqrt{2}},\, -\sqrt{2}\right)
\approx \left(+0.707106,\, +0.707106,\, -1.414213\right).
$$

## 3. The ranking check runs this on measured GPU rewards, for every preset

`test_every_preset_ranks_the_hack_last` (`tests/test_provenance.py`) loops
`for algo in ALGOS` — `grpo`, `dapo`, `cispo` — takes the **measured**
`run_task` results (not synthetic numbers), feeds them through `build_batch`
with that preset's config, and asserts

$$
\hat{A}_{\text{hack}} \;<\; \min\!\bigl(\hat{A}_{\text{triton}},\, \hat{A}_{\text{cuda}}\bigr)
\qquad \text{and} \qquad
\Bigl|\,\sum_{i} \hat{A}_i\,\Bigr| \;\approx\; 0 .
$$

That is: the hack holds the strictly lowest advantage and the group is
zero-centered. The check exists in four variants: the CUDA/Triton group
(`GPUIntegrationTests`) and one per DSL backend (TileLang, cuTile, CuTe DSL
mixins).

## 4. What is (and is not) CISPO-specific here

- **Identical advantage math.** CISPO's difference from GRPO is the *trainer's
  loss* — a policy-gradient objective weighted by a clipped, stop-gradient
  importance-sampling (IS) weight, instead of PPO's clipped surrogate:

$$
\mathcal{J}_{\text{CISPO}}(\theta)
= \mathbb{E}\!\left[\frac{1}{\sum_{i=1}^{G} |o_i|}
\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\operatorname{sg}\!\Bigl(\operatorname{clip}\!\bigl(r_{i,t}(\theta),\,
1-\epsilon^{\mathrm{IS}}_{\mathrm{low}},\,
1+\epsilon^{\mathrm{IS}}_{\mathrm{high}}\bigr)\Bigr)\,
\hat{A}_{i,t}\,
\log \pi_{\theta}\!\bigl(o_{i,t} \mid q,\, o_{i,<t}\bigr)\right],
$$

  where $r_{i,t}(\theta) = \pi_\theta(o_{i,t}\mid\cdot)\,/\,
  \pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid\cdot)$ and $\operatorname{sg}(\cdot)$
  denotes the stop-gradient (detach) operation. This repo computes the
  advantages $\hat{A}$ and the group statistics only; the loss above is
  trainer-owned and **not implemented here**. The preset records
  $\epsilon^{\mathrm{IS}}_{\mathrm{low}} = 1.0$ (no lower bound, as in
  MiniMax-M1) and $\epsilon^{\mathrm{IS}}_{\mathrm{high}} = 4.0$ (IS weight
  capped at $5$, following ms-swift's CISPO example citing ScaleRL); neither
  value affects advantage computation.
- **Preset-owned toggles that do act here:** `filter_tied_groups=True` and
  `overlong_buffer_turns=4` (environment-owned). The clip values,
  `loss_agg="token"` and `kl_coef=0.0` are recorded in the config for the
  future trainer and do not affect advantage computation at all.
- **`scripts/validate.py` produces the measured group on every GPU run:**
  after the fixtures, it rebuilds the trio through `build_batch` per preset
  and stores it under `measured_groups` in `results.json` — the JSON quoted in
  ADR 0003 and logged to W&B as `batch/<algo>/<stat>`.

## 5. The measured result (H200, run `20261107T140018Z`, `Ran 31 tests / OK`)

All four ranking checks passed under CISPO:

| Rollout | Reward $R$ | Advantage $\hat{A}$ |
|---|---|---|
| cuda | $2$ | $+0.7071062811869011$ |
| triton | $2$ | $+0.7071062811869011$ |
| TorchOp reward hack | $-1$ | $-1.4142125623738022$ |

with `adv/frac_dropped` $= 0.0$ (the group had variance, so nothing was
filtered) and `reward/mean` $= 1.0$. On real hardware, the gate keeps ranking
the cheat last under the CISPO preset — the environment's half of the
contract; the trainer's half (applying the clipped-IS-weight loss to these
advantages) remains to be built.

*Naming: the recorded run and artifacts use the historical case label
`hack`; the fixture is now named `reward-hack-torchop` (per-backend
`reward-hack-torchop-<backend>`). Historical labels are preserved verbatim.*
