# ADR 0004 — Trainer metric definitions

- **Status:** decided (2026-11-12); amended 2026-11-13 (`gpu_busy` leaves the eval denominators); emission blocked on trainer instrumentation.
- **Scope:** the eight trainer-owned reservations in [ADR 0002](0002-wandb-tracking-contract.md#trainer-owned-reservations-and-definitions).
- **Version:** `trainer-metrics-v1`. No trainer, optimizer, logging adapter,
  environment result key, reward rule or preset is implemented or changed here.

`trainer:` in ADR 0002 identifies the producer; metric keys below omit that
classification prefix. All artifact fields below are future trainer data.

## Evidence and decision boundary

[GRPO, §4.1](https://arxiv.org/html/2402.03300v1#S4.SS1)
averages token objectives within each response and then across responses.
[DAPO, §§3.1–3.3](https://arxiv.org/html/2503.14476v1#S3)
discusses entropy loss, zero-variance groups and token-level normalization.
[MiniMax-M1, §3.1](https://arxiv.org/html/2506.13585v1#S3.SS1)
clips detached IS weights while retaining token gradients. These motivate
tracking policy diversity and clipping separately from reward/advantages;
they do not specify this environment's failure cohorts or a universal
numerical collapse detector. Thresholds, reporting cohorts and workload
weights below are repo decisions, not paper-recommended hyperparameters.

Local authorities are [the presets](../utils/config.py),
[group construction](../utils/grpo.py), [reward gates](../utils/reward.py),
[stage failures](../utils/rollout_env.py), [task policy](../utils/task_policy.py)
and [the CISPO objective, §4](../CISPO_GPU_batch_ranking.md#4-what-is-and-is-not-cispo-specific-here).
Evaluation here measures authored kernel episodes, not a math benchmark's
answer accuracy. Definitions below preserve that distinction.

## Sampling unit, clock and required inputs

An episode is one agent authoring/repair trajectory ending in one terminal
candidate evaluation. Intermediate compiler/tool calls are retained in its
ledger, not counted as additional episodes or replaced by a best-ever candidate.
`B_s` contains every terminal episode collected for trainer iteration `s`,
before `build_batch` exclusions and tied-group filtering, including failures
and `gpu_busy`. Diagnostic membership does not grant loss/advantage eligibility.

`s` is the zero-based, persisted trainer `iteration`: increment once after a
collection batch is finalized, even if no optimizer update survives filtering.
An optimizer minibatch, accumulation microstep, epoch or extra pass over the
same episodes does not increment this clock or duplicate diagnostic tokens.
Evaluate entropy/ratios once at the final actor snapshot for that iteration,
after its optimizer work; with no update, use the unchanged actor. Distributed
workers sum numerators/denominators for unique episode/token IDs, not local means.

| Input | Producer and contract | Missing-data consequence |
|---|---|---|
| Episode/token IDs, exact token IDs, original contexts, tokenizer revision, generated-token ownership mask and stop/truncation metadata | Future agent sampler/episode ledger, spanning every turn | Token length, entropy and IS fraction blocked; never reconstruct from final source-file text |
| Behavior actor version and raw sampled-token log-probability at generation; final actor checkpoint/log-probabilities and full distributions at stored contexts | Future sampler and trainer diagnostic forward; behavior version is per token if collection was asynchronous | IS fraction or entropy blocked independently; never substitute reference-model log-probabilities |
| Effective `cfg.clip_low`, `clip_high`, `loss_agg`, `filter_tied_groups`, group membership, `reward/value` and returned advantage IDs | Effective config plus `build_batch` inputs/outputs | Group/clip diagnostics blocked if unknown; never recreate old configs from current defaults |
| Actual policy-turn count | Episode orchestrator; cross-check `env/turns_used` when supplied | Turn-length aggregate blocked if any count is unknown or disagrees |
| Global unscaled gradient L2 norm before gradient clipping, and whether an optimizer update was attempted | Future optimizer instrumentation; count each parameter once across shards, after data-parallel gradient reduction | Numerical collapse branch unobserved, not a zero norm |
| Terminal result, effective evaluation config/policy, runtime capability record and cohort manifest | Evaluator plus future trainer evaluation orchestrator | Apply the evaluation completeness and collapse validity rules below |

The environment currently supplies no policy-token ledger, actor entropy,
behavior log-probabilities or gradients. `env/turns_used` is agent turns,
sometimes caller-supplied/null, never generated tokens. All trainer-owned
keys remain absent from environment logging. No token metric may be inferred
from `env/wall_s`, turns, kernel dimensions or source-file character counts.

## Token mask, entropy and lengths

Let `m_it=1` only for a token actually sampled from the policy in episode `i`;
otherwise `m_it=0`. Count policy-authored kernel text, repairs and any sampled
reasoning/tool-call syntax, including a sampled EOS/stop token. Exclude prompt,
system/user text, observations, compiler output, tool results, injected wrappers,
padding and tokens after termination. Previously generated text appearing again
as later context is not counted twice. A truncated episode contributes its
actually generated prefix; failure, zero advantage, truncation, tied-group
filtering and `gpu_busy` do not erase these diagnostic tokens. This is an
ownership mask; optimizer eligibility is a separate mask.

For these diagnostics, `pi` is the actor's full-vocabulary softmax at temperature
1, before decoding temperature/top-k/top-p transformations. Record decoding
settings alongside the episode. Both behavior/current log-probabilities must
refer to this same raw distribution and context, consistent with the policy
ratio in the local CISPO objective; serving-engine warped probabilities are
not substitutes. This specifies diagnostic probabilities, not a new sampler.

Set `L_i=sum_t m_it`, `N_s=sum_i L_i`, and
`h_it=-sum_v pi_s(v|context_it) log(pi_s(v|context_it))`, with natural logarithms
and `0 log 0=0`.
Evaluate distributions/reductions only at `m_it=1` contexts; multiplying invalid
padding values by zero is not a substitute for selecting the valid tokens.

| Reserved metric | Formula and aggregation at iteration `s` | Null/absence rule |
|---|---|---|
| `train/entropy` | `sum_it m_it h_it / N_s`, nats per policy token; no reward, advantage, KL or IS weighting | Null if `N_s=0`, any ownership/context/distribution input is missing, or entropy is non-finite |
| `train/trajectory_length_tokens` | `sum_i L_i / len(B_s)`, tokens per episode, uniformly episode-weighted | Null for empty `B_s` or any unknown token ledger; an explicitly recorded zero-token episode contributes 0 |
| `train/trajectory_length_turns` | `sum_i u_i / len(B_s)`, actual agent turns per episode, uniformly episode-weighted | Null for empty `B_s` or any unknown/inconsistent `u_i`; recorded zero turns contribute 0 |

An individual zero-token episode has undefined entropy, not entropy 0; it has
no weight in the pooled token numerator/denominator and stays in the ledger
and episode-length denominator. Any invalid counted token invalidates the
whole corresponding scalar; do not drop it and report a valid-subset mean.

Entropy stays token-weighted for **all** presets. GRPO's `loss_agg="seq"` does
not rename or redefine it. The sequence-weighted alternative would be
`mean_{i:L_i>0}(sum_t m_it h_it/L_i)`; it is deliberately **not emitted as an
additional metric** under this reservation. Retain per-episode entropy sums
and counts in the trainer artifact so comparisons with sequence-normalized
loss are possible without changing the scalar's denominator. This reporting
choice is local; GRPO's sequence loss and DAPO's token loss are the precedents
for distinguishing the two weightings, not for silently switching metrics.

## IS ratio clip fraction

For the same diagnostic tokens, let
`d_it=log pi_s(token_it|context_it)-log pi_behavior_it(token_it|context_it)`
and `r_it=exp(d_it)`. The denominator is the **behavior actor**, not a KL
reference actor. Use effective config overrides, not the algorithm label:
`a=1-cfg.clip_low`, `b=1+cfg.clip_high`.

| Preset defaults | Closed interval `[a,b]` | Loss aggregation / KL |
|---|---|---|
| GRPO | `[0.8,1.2]` | `seq`; `kl_coef=0.001` |
| DAPO | `[0.8,1.28]` | `token`; `kl_coef=0` |
| CISPO | `[0,5]` | `token`; `kl_coef=0` |

`train/is_ratio_clip_fraction = sum_it m_it 1[r_it<a or r_it>b] / N_s`.
Equality at either endpoint is not clipping. Compare `d_it` with `log(a)`
and `log(b)` (`log(0)=-infinity`) to avoid exponentiation overflow. Require
finite stored/current counted-token log-probabilities and valid bounds;
otherwise the scalar is null, as it is for `N_s=0` or incomplete token data.
Do not discard invalid ratios, change the denominator, or report zero.

This is a token fraction for GRPO too, irrespective of `loss_agg="seq"`.
It measures out-of-interval ratios, **not** sign-dependent PPO surrogate
activation, a trajectory clip fraction, or the fraction of zeroed gradients.
Zero-advantage tokens remain in the denominator. CISPO clips the detached
weight; a clipped token can still contribute a gradient. For objective
auditing, retain loss-eligible episode IDs separately: the diagnostic cohort
includes excluded episodes and is not claimed to equal the optimizer cohort.

The local objective's token denominator agrees with
[MiniMax-M1 Eq. 4](https://arxiv.org/html/2506.13585v1#S3.SS1).
The upper default follows [ms-swift's official CISPO example](https://swift.readthedocs.io/en/v4.3/Instruction/GRPO/AdvancedResearch/CISPO.html#parameter-configuration):
its `epsilon_high=5.0` is an absolute cap, hence local `clip_high=4.0`.
The repo also records the [v4.4 documentation link](https://swift.readthedocs.io/en/v4.4/Instruction/GRPO/AdvancedResearch/CISPO.html);
the retrievable official v4.3 example corroborates its recorded cap convention
and attributes the setting to
[ScaleRL §A.17.2](https://arxiv.org/html/2510.13786#A17.SS2).
It is not a numeric upper setting published by MiniMax-M1.

## Collapse detector

Use the following versioned, operational detector, not a claim about the cause
of poor reward. No EMA. All thresholds are fixed within a run and identical
across presets; they are local monitoring defaults, not literature constants.

For each iteration, reconstruct original `build_batch` groups, remove only
`gpu_busy` rows, and omit groups left empty. Before tied filtering, compute:

- `Q_s`: fraction of these groups with identical final `reward/value` values,
  using exact equality as `filter_groups` does.
- `V_s`: uniformly group-weighted mean population variance of their shaped
  `reward/value` values; no sample/Bessel correction.
- `S_s`: unweighted accepted-episode fraction among their remaining rows,
  using the acceptance predicate in the evaluation section with training config.

These are null if groups/rewards/results are incomplete or their denominator
is empty. Reward shaping remains included, as in `utils/grpo.py`.

| Signal | Threshold and role |
|---|---|
| Token entropy `H_s` | Low exploration when finite `H_s <= 0.1` nats/token |
| IS drift `C_s` | Severe drift when finite clip fraction `C_s >= 0.95` |
| Stalled reward/groups | `V_s <= 1e-12`, `Q_s >= 0.95`, and `S_s <= 0.10`, all required |
| Actual `adv/frac_dropped` | Context only: GRPO disables tied filtering, so it is not an algorithm-independent collapse trigger; use counterfactual `Q_s` above |
| Advantage variance | Context only: square existing `adv/std` only with at least one returned advantage; an empty retained set is undefined despite the helper's zero-valued summary |
| Global gradient norm | Non-finite norm on an attempted update is a numerical failure; finite magnitude alone has no collapse threshold |

Confirm **behavioral collapse** at `s` when 20 consecutive collection
iterations satisfy `H_s <= 0.1` and either severe IS drift or the stalled
reward/groups condition. The drift branch requires valid `H_s,C_s`. The stalled
branch requires valid `H_s,V_s,Q_s,S_s`, a recorded usable GPU/compiler/runtime
for the workload, and no environment failure or invalid terminal result in
`B_s`. Its allowed terminal reasons are exactly the three success reasons
defined below plus `compile`, `verification`, `correctness`, `provenance` and
`max_turns`; every other string makes this branch unavailable. In particular,
`workdir`, `backend`, `timeout`, `unmeasured`, `profiling`, `timing`,
`gpu_busy`, `diagnostic_profile` and unknown reasons make the stalled branch
unavailable, not evidence of policy collapse. `profiling` is the profiler
stage exiting non-zero after verification passed (`utils/rollout_env.py`,
`finish(stage)`): a measurement failure, not a candidate failure. Candidate
`compile`, `verification`, `correctness`, `provenance` and `max_turns` failures
remain first-class reward data; a compiler capability record is necessary
because `compile` alone cannot distinguish bad source from missing tooling.
Uniform success/tied groups without IS drift therefore do not trigger collapse.

The runtime guard requires the collection device mode `cuda`,
`runtime/driver_ok=True`, `runtime/torch_cuda_build=True`, and the matching
`runtime/ok_cuda_backend`, `runtime/ok_triton_backend`, `runtime/ok_cutile`,
`runtime/ok_tilelang` or `runtime/ok_cutedsl` flag to be true for every backend
used. CUDA additionally requires
`runtime/nvcc_ok=True` and non-null `runtime/ninja`; unknown capabilities make
the stalled branch unavailable. Use the record for the assigned device;
GPU0-only metadata cannot certify a different assigned GPU without mapping.

Confirm **numerical collapse** after three consecutive collection iterations
with an attempted update and a non-finite unscaled global gradient norm.
One overflow is not sufficient. If any attempted update has a recorded
non-finite norm, this iteration qualifies for the numerical streak. Otherwise
no attempted update, a missing norm or all finite norms resets that streak;
missing is not a zero-gradient observation.

Evaluate both branches each iteration. A false/unavailable behavioral predicate
resets its 20-step streak; false/unavailable numerical evidence resets its
three-step streak. Do not bridge missing iterations. Before confirmation,
missing inputs mean unobserved monitoring, not healthy training. Log streak
lengths, signal availability and trigger branch in the trainer artifact.

`train/collapse_step` is the first **confirmation iteration**, not backdated
to the start of its window. Latch it for the run; later recovery does not
erase it. Step 0 is a valid index, never a missing-value sentinel. Persist
streaks, monitoring segment and latch in checkpoints; resumes retain them.
A changed tokenizer, diagnostic probability convention or workload/runtime
protocol starts a recorded monitoring segment and resets streaks, not the
earliest confirmed latch. The summary is null/absent until confirmation;
absent when no detector is implemented. Unobserved/insufficient monitoring
must be disclosed in the artifact rather than described as "never collapsed".

DAPO's entropy/group discussion motivates these signals, not these cutoffs.
The runtime guard follows this repo's coarse stage-failure taxonomy; neither
low reward nor a high dropped-group fraction alone diagnoses trainer collapse.
The 0.1-nat floor targets near-determinism; `1e-12` is the square of the
group normalizer's `1e-6` stabilizer. The 95%/10% cutoffs select widespread
loss of signal with little accepted execution. Twenty iterations and three
numerical failures require persistence instead of a single noisy batch/overflow.
These are operational choices to validate during future training, not measured
collapse thresholds or an automatic stop/restart policy.

## Evaluation cohort and weights

Freeze a held-out evaluation manifest before comparing checkpoints/algorithms:
task/case/backend cells, source/verification/provenance policy, effective
thresholds and `speedup_margin`, timing-method version, requested device mode,
episode budgets, attempt IDs/seeds, and decoding settings. No training-group
advantages or DAPO sampling frequencies become evaluation weights.
Attempt IDs identify manifest slots and stay the same across algorithm runs;
run-specific episode IDs distinguish executions of those slots.

One attempted episode is one dispatched, terminally accounted manifest slot,
including a zero-token failure. Record each terminal result once. Do not select
the best candidate/retry or discard failed attempts. Additional retries require
a new manifest revision preserving the old attempts; compare only compatible
manifests. Until every declared slot has a terminal result and every declared
cell has at least one attempt, the cohort is incomplete and all three eval
aggregates are null. Missing outcomes are not silently invented failures.

Let `J` be the number of declared tasks, `K_j` the number of case/backend cells
of task `j`, and `n_jk` the attempted episodes in cell `(j,k)` whose terminal
reason is not `gpu_busy`. Each such episode has fixed weight
`w_i=1/(J K_j n_jk)`, so weights sum to 1: uniform tasks, uniform cells within
a task, uniform attempts within a cell. A `gpu_busy` episode has weight 0 and
is outside every eval denominator. If every attempt in a declared cell is
`gpu_busy` (`n_jk=0`), the cohort is incomplete and all three aggregates are
null until a new manifest revision re-runs that cell. More catalog cases,
`cfg.group_size` or resampling do not increase a task's weight. Use these same
weights for all checkpoints; changed assignments require a different cohort ID.

| Denominator policy | Decision |
|---|---|
| All attempted episodes except `gpu_busy`, failures count against rates | **Choose** for both rates: preserves end-to-end reliability and avoids survivor selection. `gpu_busy` is GPU-lock contention between rollouts, not a policy outcome; it is already excluded from training advantages (`utils/grpo.py`) and is reported as its own fraction |
| Measured-only rates, accompanied by existing `reward/frac_*` | Reject as the primary rates: CPU/unavailable/timeout rows disappear. Existing fractions cover provenance/max-turns, not the full taxonomy; training-batch fractions are not eval-cohort fractions |
| Per-reason breakdown | Require alongside the aggregates in the trainer eval artifact; complements rather than replaces a fixed denominator |

Define `P_i=1` only when terminal `reward/reason` is one of `correct`,
`faster_than_eager`, `faster_than_both`; `env/verify_ok=True`; all three
`env/t_eager_us`, `env/t_compile_us`, `env/t_gen_us` are finite and strictly
positive; and the manifest's provenance gate passes. Stock uses finite
`env/provenance` in `[cfg.provenance_threshold,1]`. Progressive tasks additionally
require the recorded task policy, `task/provenance_authored` at its recorded
threshold, zero `env/provenance_library`, and no `task/policy_violation`.
Unknown policy/threshold or inconsistent/missing success evidence gives `P_i=0`
and a data-quality flag; retain the original reason. Missing terminal identity/
reason makes the cohort incomplete. Diagnostic timing passes cannot qualify.
This is end-to-end accepted execution, not verification-only accuracy, and is
independent of shaped `reward/value` (which can be zero on successful episodes).
Set `F_i=0` when `P_i=0`; otherwise set
`F_i=1[t_gen_i < (1-cfg.speedup_margin_i) t_compile_i]`.
Do not evaluate a timing comparison on failed/unmeasured rows.

| Metric | Formula | Null rule |
|---|---|---|
| `eval/pass_rate` | `sum_i w_i P_i` | Empty/incomplete manifest or unknown weights/protocol |
| `eval/faster_rate_vs_compile` | `sum_i w_i F_i` | Same completeness rule; failed/unmeasured episodes contribute 0, not a smaller denominator (only `gpu_busy` has weight 0) |
| `eval/geomean_speedup_vs_compile` | For `A={i:P_i=1}`, `c=sum_{i in A}w_i`; `exp(sum_{i in A} w_i (log t_compile_i-log t_gen_i)/c)` | Same completeness rule, or `c=0`; non-representable/non-finite final value is null with a data-quality flag |

The faster-rate boundary is strict and uses the recorded effective margin
(default 0.05), matching `utils/reward.py`. Do not count only
`faster_than_both`: a `correct` episode can beat compile without beating eager.
Example: eager 50, compile 100, generated 70 microseconds is `correct` yet
qualifies as faster than compile at margin 0.05. Geomean includes accepted
slowdowns too; never select only faster episodes or average speedups arithmetically.
Use log differences for numerical stability. Zero, negative, missing or
non-finite times never receive a ratio, epsilon replacement, speedup 0 or 1.
Require a finite, strictly positive final geomean; overflow or underflow to
zero makes it null with a data-quality flag, not a clamped speedup.

| Terminal outcome | Rate contribution / geomean |
|---|---|
| Accepted success reasons with valid evidence | Pass 1; faster predicate above; positive measured ratio enters geomean |
| `compile`, `verification`, `correctness`, `provenance`, `max_turns` | Pass/faster 0. Provenance failure may have positive timings, but they cannot claim authored performance; exclude from geomean |
| `unmeasured`, including CPU/interpreter or provenance None | Pass/faster 0 as measurement-unavailable, not correctness failure; geomean excluded |
| `timeout` | Pass/faster 0 as budget failure, separately counted; no imputed completion time/ratio |
| `workdir`, `backend`, `profiling`, `timing`, `diagnostic_profile`, unknown reasons | Pass/faster 0, preserve exact reason and data-quality details; geomean excluded even if stale/partial timings exist. `profiling`, `timing` and `diagnostic_profile` are measurement failures after verification passed; report them as such, not as correctness failures |
| `gpu_busy` | Excluded: weight 0, in no rate denominator and not in the geomean. Report its raw and weighted fraction of attempted episodes in the per-reason breakdown |

`gpu_busy` is excluded from **training advantages** and from **eval
denominators** alike, so GPU contention can neither lower a pass rate nor
inflate one; its frequency stays visible in the per-reason breakdown, with the
attempted-episode count it was drawn from. An all-unmeasured completed cohort has
both rates 0 and geomean null. It is not evidence of zero numerical correctness.

The geomean is explicitly conditional on accepted measurements, unlike the
two rates. Require the artifact's accepted attempt-ID mask, raw and weighted
coverage `c`, per-task/cell retained weights, and raw/weighted fractions for
**every exact** `reward/reason` (include zero bins for known reasons, preserve
new strings). Include verification-only counts separately. Failures remain
visible even when excluded from the geomean; existing `reward/frac_*` are
insufficient substitutes. Cross-algorithm geomean comparisons require identical
accepted attempt/cell masks and fixed weights; otherwise mark cohorts different,
report the two all-attempted rates, and do not rank the conditional geomeans.

The manifest, ledgers, coverage, per-reason tables, signal validity and detector
state are future trainer artifacts, not additional reserved scalar names or
environment outputs. These evaluation weights/cohorts are repo choices:
GRPO/DAPO/CISPO objectives do not prescribe a CUDA compile/provenance/availability
denominator. No current environment logger may fabricate these trainer metrics.
