# CISPO GPU-batch ranking in this codebase

> Draft: not linked from the README. Live fixtures use `reward-hack-torchop` and `reward-hack-torchop-<backend>`; historical artifacts retain their recorded labels.

*Explainer note (2026-11-07). Companion to [Architecture-decision/0001-cispo-preset.md](Architecture-decision/0001-cispo-preset.md) (the decision) and [Architecture-decision/0003-cispo-gpu-validation.md](Architecture-decision/0003-cispo-gpu-validation.md) (the measured evidence).*

"CISPO GPU-batch ranking" is the same group-relative advantage machinery the
environment already uses for GRPO/DAPO — CISPO changes nothing in how
advantages are computed. What the codebase validates is that, under the CISPO
preset, the reward-hacking rollout still ranks last in a group **measured on
real hardware**. The pipeline, end to end:

## 1. Base reward and final rollout reward

Every member of the group is a full `run_task` evaluation on the GPU
(compile → verify → profile). `compute_reward` in
[`utils/reward.py`](utils/reward.py) assigns a discrete base reward.
Let $g_i$ denote passage of the correctness, provenance, valid-timing and
turn-cap gates for rollout $i$. With margin $\delta = 0.05$ by default, define

$$
E_i := t_{\mathrm{gen},i} < (1-\delta)t_{\mathrm{eager},i},
\qquad
C_i := t_{\mathrm{gen},i} < (1-\delta)t_{\mathrm{compile},i}.
$$

The mutually exclusive milestone cases are

$$
R_i^{\mathrm{base}} =
\begin{cases}
-1, & \neg g_i,\\
3, & g_i \land E_i \land C_i,\\
2, & g_i \land E_i \land \neg C_i,\\
1, & g_i \land \neg E_i.
\end{cases}
$$

Stage failures and timeouts in `run_task` also return $-1$. The inequalities
are strict: matching a margin boundary does not earn the higher milestone.
The final reward is $R_i = R_i^{\mathrm{base}} - p_i$, where $p_i$ is the
optional overlong-turn penalty. For a successful rollout with known turn
count $u_i \leq T$, maximum turns $T$, and buffer $B > 0$,

$$
p_i = \max\!\left(0,\frac{u_i-(T-B)}{B}\right).
$$

The penalty is zero for failures, an unknown turn count, or $B=0$.
DAPO and CISPO use $B=4$; GRPO uses $B=0$. Thus the base reward is discrete,
but the final reward supplied to `build_batch` need not be an integer.

In the measured H200 group: honest CUDA → $R=2$, honest Triton → $R=2$,
TorchOp reward hack → $R=-1$ (provenance $0.0$ against the $0.90$ gate).
Each used one turn, so the overlong penalty was zero.

## 2. `build_batch` groups the rollouts and standardizes rewards within each group

[`utils/grpo.py`](utils/grpo.py) is Torch-free and algorithm-agnostic: it
reads configuration fields rather than branching on the algorithm name. It:

- **buckets rollouts by `task_id`** and splits them into whole groups of
  $G$, given by `cfg.group_size` (incomplete groups are errors). For a group
  $\mathcal{G} = \{\tau_1, \dots, \tau_G\}$ with rewards
  $\{R_1, \dots, R_G\}$:
- **applies the tied-group filter** `filter_groups(groups, cfg.filter_tied_groups)`
  — the preset-controlled filtering behavior in the batch builder. CISPO and
  DAPO set `filter_tied_groups=True`, so the group is **dropped** whenever

$$
\min_{1 \leq i \leq G} R_i \;=\; \max_{1 \leq i \leq G} R_i
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

  using the **population** standard deviation. In exact arithmetic,
  $\sum_i \hat{A}_i = 0$; tests allow floating-point tolerance. The stabilizer
  makes the population standard deviation of a non-tied advantage group
  $\sigma_{\mathcal{G}}/(\sigma_{\mathcal{G}}+\varepsilon)$, slightly below
  one. With `adv_norm="none"`, the code instead uses $R_i-\mu_{\mathcal{G}}$
  without dividing by the standard deviation.

For the measured rewards $(2,\,2,\,-1)$: $\mu_{\mathcal{G}} = 1$,
$\sigma_{\mathcal{G}} = \sqrt{2}$. Including the implementation's stabilizer,

$$
\bigl(\hat{A}_{\text{cuda}},\, \hat{A}_{\text{triton}},\, \hat{A}_{\text{hack}}\bigr)
= \left(
\frac{1}{\sqrt{2}+10^{-6}},\,
\frac{1}{\sqrt{2}+10^{-6}},\,
\frac{-2}{\sqrt{2}+10^{-6}}
\right)
\approx (0.7071062811869011,\;0.7071062811869011,\;-1.4142125623738022).
$$

The frequently quoted values $(1/\sqrt{2},1/\sqrt{2},-\sqrt{2})$ apply only
in the limit $\varepsilon \to 0$, not exactly at $\varepsilon=10^{-6}$:

$$
\lim_{\varepsilon\to0^+}
\bigl(\hat{A}_{\text{cuda}},\hat{A}_{\text{triton}},\hat{A}_{\text{hack}}\bigr)
= (1/\sqrt{2},\,1/\sqrt{2},\,-\sqrt{2}).
$$

## 3. The ranking check runs this on measured GPU rewards, for every preset

`test_every_preset_ranks_the_hack_last` in
[`tests/test_provenance.py`](tests/test_provenance.py) loops
`for algo in ALGOS` — `grpo`, `dapo`, `cispo`. The GPU fixtures are evaluated
once using the GRPO preset at one turn; each preset is then passed to
`build_batch` over those same measured rewards. This is not a separate
CISPO rollout, optimizer step, or training run.

For the CUDA/Triton trio, the test asserts

$$
\hat{A}_{\text{hack}} \;<\; \min\!\bigl(\hat{A}_{\text{triton}},\, \hat{A}_{\text{cuda}}\bigr)
\qquad \text{and} \qquad
\Bigl|\,\sum_{i} \hat{A}_i\,\Bigr| \;\approx\; 0 .
$$

Three additional checks compare an honest rollout with its backend-specific
Torch-op reward-hack counterpart, one pair each for TileLang, cuTile and
CuTe DSL. They assert the lower advantage of the reward hack; unlike the
trio check, these pair checks do not explicitly assert a zero sum.

**Scope of the ranking evidence:** it establishes relative ordering for
these non-tied, one-turn groups under each batch preset. It does not exercise
the tied-group drop branch or the overlong penalty, and it does not test the
CISPO loss.

Those behaviors have separate checks:

| Behavior | Test | Evidence type |
|---|---|---|
| Keep tied groups for GRPO; drop them for DAPO/CISPO | `GroupTests.test_identical_groups_under_every_preset` in `tests/test_config_reward_grpo.py` | Synthetic rewards; pure-Python batch logic |
| CISPO/DAPO penalty of 0.75 at turn 19, unchanged base reward | `RolloutProtocolTests.test_provenance_gate_and_overlong_via_entry_point` in `tests/test_rollout_env.py` | Mocked stage outputs; rollout/reward wiring |
| DAPO overlong penalty on a real GPU rollout | `GPUIntegrationTests.test_dapo_overlong_base_unchanged` in `tests/test_provenance.py` | Separate GPU test; not a CISPO-specific shaping test |

## 4. What is (and is not) CISPO-specific here

- **Identical advantage math.** CISPO's difference from GRPO is the *trainer's
  objective* — a policy-gradient objective weighted by a clipped, stop-gradient
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
  denotes the stop-gradient (detach) operation. Here $q$ is the task prompt,
  $o_i$ is a sampled output sequence, and $t$ indexes its tokens. The
  expectation is over prompts and old-policy samples. The objective is
  maximized; a minimization loss would be $-\mathcal{J}_{\mathrm{CISPO}}$.
  A trainer would broadcast the sequence advantage $\hat{A}_i$ to its valid
  tokens. This repo does not collect token log-probabilities or implement
  token masking, this objective, or an optimizer.

  The preset records $\epsilon^{\mathrm{IS}}_{\mathrm{low}} = 1.0$
  (`clip_low=1.0`), so the lower clipping threshold is zero, non-binding for
  a nonnegative importance ratio. This matches the paper's choice of no
  effective lower bound; it is not a published numeric setting from the
  paper. The preset uses $\epsilon^{\mathrm{IS}}_{\mathrm{high}} = 4.0$
  (`clip_high=4.0`), so the upper weight cap is $5$, following ms-swift's
  CISPO example citing ScaleRL. MiniMax-M1 gives no numeric upper setting.
  Neither clip value affects the environment's advantage computation.
- **Preset-owned toggles that do act here:** `filter_tied_groups=True` and
  `overlong_buffer_turns=4` (environment-owned). The clip values,
  `loss_agg="token"` and `kl_coef=0.0` are recorded in the config for the
  future trainer and do not affect advantage computation at all.
- **`scripts/validate.py` conditionally records measured groups:**
  after the fixtures, it rebuilds the trio through `build_batch` per preset
  only when all three required rollouts have non-null provenance. It stores
  the groups under `measured_groups` in `results.json` — the JSON quoted in
  ADR 0003 and logged to W&B as `batch/<algo>/<stat>`. It also writes a
  separately labelled synthetic arithmetic demonstration; those numbers
  must not be presented as GPU measurements.

### Decision 1: keep `clip_high=4.0` (IS-weight cap 5.0)

The cap is an implementation default supported by the following source chain,
not a numeric hyperparameter published by MiniMax-M1:

- [MiniMax-M1](https://arxiv.org/html/2506.13585v1) defines the weight as
  $\operatorname{clip}(r,1-\epsilon_{\mathrm{low}}^{\mathrm{IS}},1+\epsilon_{\mathrm{high}}^{\mathrm{IS}})$.
  It states that no lower bound was imposed, by setting the lower epsilon to
  a large value, and that only the upper epsilon was tuned. It does not
  publish a numeric upper setting.
- [ms-swift's CISPO example](https://swift.readthedocs.io/en/v4.4/Instruction/GRPO/AdvancedResearch/CISPO.html)
  uses `--loss_type cispo --epsilon_high 5.0`. Its weight is
  $\min(r,\texttt{epsilon\_high})$: here `epsilon_high` is an absolute cap,
  not an offset from one. The documentation attributes that value to ScaleRL
  because MiniMax-M1 does not provide specific parameter settings.
- [ScaleRL, Appendix A.17.2](https://arxiv.org/html/2510.13786) varies the upper
  CISPO clipping threshold while keeping the lower threshold at zero and
  reports CISPO to be "largely insensitive to this hyperparameter" in its
  experiments. That is empirical evidence for that setup, not a guarantee
  that the threshold is irrelevant for every trainer or workload.

In this repository's offset convention,

$$
1-\texttt{clip\_low}=0,\qquad
1+\texttt{clip\_high}=5
\quad\Longleftrightarrow\quad
\texttt{clip\_low}=1,\quad\texttt{clip\_high}=4.
$$

For nonnegative importance ratios, these defaults give
$w(r)=\operatorname{clip}(r,0,5)=\min(r,5)$: there is no effective lower
clipping bound. The current preset in [`utils/config.py`](utils/config.py) is

```python
clip_low=1.0, clip_high=4.0, loss_agg="token", kl_coef=0.0,
filter_tied_groups=True, overlong_buffer_turns=4
```

Configuration validation accepts `clip_low=1.0` only for `algo="cispo"`;
GRPO and DAPO require `clip_low < 1`, and CISPO rejects `clip_low=1.5`.
Presets are applied before CLI overrides: `--algo cispo --clip-high 2.0`
selects an upper weight cap of 3.0, not 2.0. Keep cap 5.0 as the documented
default unless trainer-side IS-ratio observations motivate another value.

### Decision 2: keep the CISPO loss trainer-owned

Clip ranges, loss aggregation and KL settings are trainer-owned for GRPO,
DAPO and CISPO alike. The environment owns group-relative advantages,
tied-group filtering and overlong-turn reward shaping. It has no trainer.
At the default CISPO bounds, the objective's per-token term is

$$
\operatorname{sg}\!\left(\min(r_{i,t}(\theta),5)\right)
\hat{A}_{i,t}\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}),
\qquad
r_{i,t}(\theta)=\exp\!\left(\log\pi_\theta-\log\pi_{\theta_{\mathrm{old}}}\right),
$$

where the two log-probabilities in the ratio refer to the same token and
conditioning context. Those policy log-probabilities and their gradients
belong to the trainer; the evaluator neither receives them nor computes
this loss. The advantage and reward results therefore remain unchanged by
the clip/KL configuration fields.

**Proposed follow-up, not implemented:** a Torch-free scalar reference helper
in `utils/cispo.py`, with signature
`clipped_is_weight(log_ratio, clip_low, clip_high) -> float`, using only
`math`, and tests for known ratios, clipping boundaries and the no-effective-
lower-bound case. It would make the weight convention executable without
adding a trainer. A scalar helper cannot implement autograd's stop-gradient;
the trainer must still detach the weight. Keep any such helper outside
`reward.py` and `rollout_env.py`, with no reward or result-schema changes.

## 5. The measured result (H200, run `20261107T140018Z`, `Ran 31 tests / OK`)

All four ranking checks passed under CISPO:

| Rollout | Reward $R$ | Advantage $\hat{A}$ |
|---|---|---|
| cuda | $2$ | $+0.7071062811869011$ |
| triton | $2$ | $+0.7071062811869011$ |
| TorchOp reward hack | $-1$ | $-1.4142125623738022$ |

with `adv/frac_dropped` $= 0.0$ (the group had variance, so nothing was
filtered) and `reward/mean` $= 1.0$. This historical run demonstrates that
measured provenance-gated rewards place the ordinary Torch-op reward hack
last after CISPO-configured batch processing. It does not by itself
validate tied-group removal, CISPO overlong shaping, the clipped-IS
objective, or general resistance to adversarial reward hacking.

Source evidence: [structured CISPO validation](artifacts/cispo-validation-20261107T140018Z.json),
[measured results](artifacts/results-20261107T140018Z.json), and
[GPU suite log](artifacts/gpu_tests.stderr-20261107T140018Z.log).
Algorithm and clip-setting sources: [MiniMax-M1](https://arxiv.org/html/2506.13585v1),
[ms-swift CISPO](https://swift.readthedocs.io/en/v4.4/Instruction/GRPO/AdvancedResearch/CISPO.html),
and [ScaleRL](https://arxiv.org/html/2510.13786).

*Naming: `hack` in this run's JSON is the historical case label. Current
fixtures use `reward-hack-torchop` (per-backend
`reward-hack-torchop-<backend>`). Historical artifacts keep their recorded names.*
