# ADR 0002 — W&B tracking contract

Status: implemented; trainer-owned measurement definitions remain reserved.
This filename follows the task request; the CISPO GPU validation evidence is
recorded separately in 0003-cispo-gpu-validation.md.

## Decision and run shape

Choose (a): heterogeneous validation cases live only in the `rollouts` table.
One validation invocation is one W&B run; its batch comparison has a single
snapshot at `iteration=0`, with `batch/<algo>/<stat>` metric keys. Never log case
scalars as steps. Tables are case-level diagnostic records; `run.summary` has
only explicit aggregate scalars (plus SDK table/artifact references).
`define_metric("*", step_metric="iteration", summary="none")` disables default
last-case summaries; resumed backfills clear stale case summary fields.
Historical metric history cannot be erased by this adapter; a rerun replaces
its explicit summary and adds the corrected tables. Dry-run never contacts W&B.
No remote W&B run is changed by this implementation task.

Training uses one run per (algo, seed), group=algo, with caller-supplied integer
iterations. `log_batch` calls the unchanged build_batch and logs its ten stats
under their original adv/, reward/, turns/ keys once per training iteration.
`log_rollout` is a standalone/case-table hook, not a training-iteration counter.
It has no batch advantage stats until a complete group is supplied to log_batch.
Reuse the returned run across iterations; passing run=None intentionally creates
a new run. The trainer owns checkpoint/resume identity and iteration numbering.
All means/rates in batch stats retain build_batch's population and percentile
semantics. Explicit run summaries aggregate all collected case rows; no default
last-case reward/value, reason, provenance, task or backend is permitted.
Sweep objective is aggregate `reward_mean`, replacing per-case reward/value.

## Minimum evidence (must track)

- Full cfg/* plus per-case and per-group overrides: reproduce actual evaluation
  and grouping, rather than current defaults. Old artifacts stay unknown.
- Runtime GPU UUID, name/CC, driver, toolkit/compiler, package/Python versions
  and ok_* flags: identify device and execution capability.
- meta/backend and meta/task_id: associate each result with its implementation
  and workload; meta/algo and seed distinguish experimental conditions.
- Candidate model_new.py SHA-256 in the rollout table (also Weave): identify
  the candidate even when tracing is disabled; kernel files remain artifacts.
- Classifier rule-set ID from utils/provenance.py: attribution semantics affect
  reward. New artifacts persist it; historical unknown IDs are null, never
  relabelled with the current classifier. Source revision/hashes belong in raw
  evidence artifacts where available; SHA of the candidate alone cannot
  reproduce imported kernel code without its source artifact.
- reward/value, base, reason and penalty: interpret the milestone and shaping.
- env/verify_ok, env/provenance and four shares: establish correctness and why
  attribution passed/failed. Shares are total-device-time fractions, whereas
  provenance excludes memory time. They are diagnostic decompositions, not
  additional reward inputs.
- env/t_eager_us, env/t_compile_us, env/t_gen_us and speedup_vs_compile:
  reproduce the actual reward comparisons; these are profiler sums, not wall
  latency. env/turns_used and wall_s describe budget use.
- The ten build_batch statistics for grouped training or measured backfill:
  show group filtering, reward mix, advantage scale and rollout turn lengths.
  A standalone rollout has no valid batch statistics; do not invent them.
- Raw results/config/runtime/source evidence artifact: audit the above.

A run with missing minimum fields is incomplete evidence, not reproducible
from W&B alone. tracking/* availability flags expose historical omissions.
The requested 20261106T185236Z dry-run cannot acquire cfg, hashes or classifier
identity from current defaults. Persisting config fixes future runs, not old
measurements. Its profile shares cannot be borrowed from the independent honest
name-recorder fixture, especially for hacks, overlong or timeout cases.

## Optional evidence and boundaries

bench/*, kernel rows, traces, Nsight output, raw latency samples and Weave source
text are optional diagnostics. Never use bench timings or artifacts to recompute
reward. log/* stage tails go only to Weave and raw artifact files; they are
excluded from W&B scalar history, tables and summary. Artifact paths are dropped
from scalar metrics; referenced files are artifact data when available.
Synthetic group demo values stay in labelled raw artifacts, not metric history.

The current default result schema has 28 keys, including the four diagnostic
`env/provenance_*` shares and six `meta/gpu_*` identity/lock fields. An opt-in
`profile_out` dictionary captures the same profiling pass in the trusted parent
without SDK/Torch imports or an extra GPU pass. GPU identity and lock wait never
enter `compute_reward`; only the approved `gpu_busy` environment failure is
excluded within its original group before advantage normalization/filtering.
Validation artifacts now persist those profiles, full base/case/group configs,
candidate hashes and rule-set ID. Live evidence artifacts include candidate files,
trusted utils sources and requirement pins. The name-recorder header carries full config
and explicit compile/profile-only scope: its reward/verification settings are
recorded defaults, not a claim that reward or verification ran in that tool.

W&B API reference: [Run.define_metric source](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py).
The documented summary="none" behavior is tested with a fake SDK, not a live
service migration. Full backfill is opt-in and is not executed here.
This describes the initial implementation stage. The subsequent explicitly
approved live validation and five-entry migration passed service readback and
user UI inspection; see `artifacts/tracking-live-check-20261108T142009Z/report.json`
and `artifacts/tracking-backfill-20261108T144457Z/report.json`. Legacy history
was preserved; current summaries and latest table snapshots follow this contract.

## Trainer-owned reservations and definitions

[ADR 0004](0004-trainer-metric-definitions.md) defines masking, denominators,
collapse detection, eval cohorts/weights and their instrumentation dependencies.
The trainer rows below remain reservations: environment logging never emits
fabricated trainer metrics. `env/turns_used` is agent turns, not policy tokens;
token counts/log-probabilities require the future episode ledger and trainer.
No trainer or optimizer is implemented by these definitions.

## Machine-checked classification table

New `profile/*` counters, eager names, order/trace metadata and `timing_reason`
are retained in raw artifacts, without adding per-case W&B metric steps.
`timing_measured`, existing timings and shares retain their table projections.

Each source:key is assigned exactly one category. Source distinguishes a raw
record from its projected tracking row; rows are not extra evaluator keys.
`*` covers dynamic event names, case labels, package names or artifact paths.
Units and null rules apply to each expansion. Missing optional keys are absent,
not zero. Raw benchmark samples are artifacts. Setting `CUDA_AGENT_BENCH_SAMPLES_DIR`
saves the sampler output (enabled in the Modal test runner).

| Source:key | Category | Namespace | Type | Unit | Null/absence condition | Role | Owner |
|---|---|---|---|---|---|---|---|
| `cfg:cfg/backend` | config | cfg | str | 1 | never | config | environment |
| `cfg:cfg/iters` | config | cfg | int | 1 | never | config | environment |
| `cfg:cfg/exact_integer_outputs` | config | cfg | bool | 1 | not recorded in historical configs | verification/config; forced true only by progressive suite | environment |
| `cfg:cfg/relative_error_bounds` | config | cfg | bool | 1 | not recorded in historical configs | Stage 6 relative verification/config; forced true only for Stage 6 arithmetic | environment |
| `cfg:cfg/gpu_lock` | config | cfg | bool | 1 | not recorded in historical configs | GPU measurement serialization/config | environment |
| `cfg:cfg/gpu_lock_timeout_s` | config | cfg | float | s | not recorded in historical configs | GPU wait cap/config | environment |
| `cfg:cfg/benchmark` | config | cfg | bool | 1 | never | config | environment |
| `cfg:cfg/bench_warmup_iters` | config | cfg | int | 1 | never | config | environment |
| `cfg:cfg/bench_rounds` | config | cfg | int | 1 | never | config | environment |
| `cfg:cfg/bench_iters_per_round` | config | cfg | int | 1 | never | config | environment |
| `cfg:cfg/cuda_graphs` | config | cfg | bool | 1 | never | config | environment |
| `cfg:cfg/diagnostics` | config | cfg | bool | 1 | never | config | environment |
| `cfg:cfg/timeout_s` | config | cfg | float | s | never | config | environment |
| `cfg:cfg/provenance_threshold` | config | cfg | float | 1 | never | reward/config | environment |
| `cfg:cfg/speedup_margin` | config | cfg | float | 1 | never | reward/config | environment |
| `cfg:cfg/max_turns` | config | cfg | int | turns | never | reward/config | environment |
| `cfg:cfg/max_context_tokens` | config | cfg | int | tokens | never | config | environment |
| `cfg:cfg/algo` | config | cfg | str | 1 | never | config | environment |
| `cfg:cfg/group_size` | config | cfg | int | 1 | never | config | environment |
| `cfg:cfg/adv_norm` | config | cfg | str | 1 | never | config | environment |
| `cfg:cfg/filter_tied_groups` | config | cfg | bool | 1 | never | config | environment |
| `cfg:cfg/overlong_buffer_turns` | config | cfg | int | turns | never | reward/config | environment |
| `cfg:cfg/clip_low` | config | cfg | float | 1 | never | config | environment |
| `cfg:cfg/clip_high` | config | cfg | float | 1 | never | config | environment |
| `cfg:cfg/loss_agg` | config | cfg | str | 1 | never | config | environment |
| `cfg:cfg/kl_coef` | config | cfg | float | 1 | never | config | environment |
| `cfg:cfg/seed` | config | cfg | int | 1 | never | config | environment |
| `cfg:cfg/run_name` | config | cfg | str | 1 | never | config | environment |
| `cfg:cfg/tags` | config | cfg | list[str] | 1 | never | config | environment |
| `result:env/verify_ok` | table | env | bool | 1 | never | reward/input | environment |
| `result:env/t_eager_us` | table | env | float | us | CPU/failure | reward/input | environment |
| `result:env/t_compile_us` | table | env | float | us | CPU/failure | reward/input | environment |
| `result:env/t_gen_us` | table | env | float | us | CPU/failure | reward/input | environment |
| `result:env/provenance` | table | env | float | fraction | CPU/failure | reward/input | environment |
| `result:env/provenance_agent` | table | env | float | fraction of total device time | CPU/failure/unmeasured | diagnostic | environment |
| `result:env/provenance_library` | table | env | float | fraction of total device time | CPU/failure/unmeasured | diagnostic | environment |
| `result:env/provenance_torch` | table | env | float | fraction of total device time | CPU/failure/unmeasured | diagnostic | environment |
| `result:env/provenance_memory` | table | env | float | fraction of total device time | CPU/failure/unmeasured | diagnostic | environment |
| `result:env/speedup_vs_compile` | table | env | float | ratio | CPU/failure | derived reward diagnostic | environment |
| `result:env/wall_s` | table | env | float | s | never | deadline diagnostic | environment |
| `result:env/turns_used` | table | env | int | turns | caller omitted | reward/input | environment |
| `result:reward/value` | table | reward | float | reward | never | reward/output | environment |
| `result:reward/base` | table | reward | int | reward | never | reward/output | environment |
| `result:reward/reason` | table | reward | str | 1 | never | reward/output | environment |
| `result:reward/overlong_penalty` | table | reward | float | reward | never | reward/output | environment |
| `result:meta/algo` | table | meta | str | 1 | never | identity | environment |
| `result:meta/backend` | table | meta | str | 1 | never | identity | environment |
| `result:meta/seed` | table | meta | int | 1 | never | identity | environment |
| `result:meta/task_id` | table | meta | str | 1 | never | identity | environment |
| `result:meta/gpu_uuid` | table | meta | str | UUID | CPU/unavailable/not reached | measured identity | environment |
| `result:meta/gpu_name` | table | meta | str | device name | CPU/unavailable/not reached | diagnostic | environment |
| `result:meta/gpu_uuid_reason` | table | meta | str | reason | UUID resolved | diagnostic | environment |
| `result:meta/gpu_lock_key` | table | meta | str | lock identity | CPU/not reached | diagnostic | environment |
| `result:meta/gpu_lock_wait_s` | table | meta | float | s | never; zero uncontended/no lock | diagnostic; excluded from deadline only | environment |
| `result:meta/gpu_lock` | table | meta | str | lock status | never | diagnostic | environment |
| `result:log/compile_tail` | weave | log | str | text | empty if none | diagnostic | environment |
| `result:log/stderr_tail` | weave | log | str | text | empty if none | diagnostic | environment |
| `result:artifact/*` | artifact | artifact | str | path | artifact unavailable | diagnostic | environment |
| `profile:provenance/agent` | table | provenance | float | fraction of total device time | CPU/failure | diagnostic | environment |
| `row:env/provenance_agent` | table | env | str | fraction of total device time | raw profile missing/CPU/failure | diagnostic | environment |
| `profile:provenance/library` | table | provenance | float | fraction of total device time | CPU/failure | diagnostic | environment |
| `row:env/provenance_library` | table | env | str | fraction of total device time | raw profile missing/CPU/failure | diagnostic | environment |
| `profile:provenance/torch` | table | provenance | float | fraction of total device time | CPU/failure | diagnostic | environment |
| `row:env/provenance_torch` | table | env | str | fraction of total device time | raw profile missing/CPU/failure | diagnostic | environment |
| `profile:provenance/memory` | table | provenance | float | fraction of total device time | CPU/failure | diagnostic | environment |
| `row:env/provenance_memory` | table | env | str | fraction of total device time | raw profile missing/CPU/failure | diagnostic | environment |
| `profile:t_eager_us` | table | t_eager_us | float | us | CPU/failure | reward/input | environment |
| `profile:t_compile_us` | table | t_compile_us | float | us | CPU/failure | reward/input | environment |
| `profile:t_gen_us` | table | t_gen_us | float | us | CPU/failure | reward/input | environment |
| `profile:provenance` | table | provenance | float | fraction | CPU/failure | reward/input | environment |
| `results:stage_wall_s/*/*` | artifact | stage_wall_s | float | s | opt-in stage timing disabled/not executed | diagnostic | environment |
| `profile:timing_reason` | artifact | profile | str | 1 | null on successful capture | diagnostic | environment |
| `profile:profile/*_events` | artifact | profile | int | CUDA events | absent on CPU | diagnostic | environment |
| `profile:profile/eager_kernel_names` | artifact | profile | str (JSON list) | 1 | absent on CPU | diagnostic | environment |
| `profile:profile/*_kernel_samples` | artifact | profile | str (JSON list of [name, calls, total_us]; per-event list in the kernel_samples_path sidecar since 2026-11-09) | us per CUDA event; non-finite durations encoded as strings | absent on CPU | diagnostic | environment |
| `profile:profile/kernel_samples_path` | artifact | profile | str | path to the per-event JSON sidecar | not written (no samples dir or OSError); absent on CPU and before 2026-11-09 | diagnostic | environment |
| `profile:profile/order` | artifact | profile | str | 1 | absent on CPU | diagnostic | environment |
| `profile:profile/timing_method` | artifact | profile | str | versioned capture conditioning | absent historically/CPU | diagnostic | environment |
| `profile:profile/primer_event_counts` | artifact | profile | str (JSON list) | CUDA events per intentional empty conditioning capture | absent historically/CPU | diagnostic | environment |
| `profile:profile/diagnostic` | artifact | profile | bool | 1 | absent on CPU | diagnostic | environment |
| `profile:profile/*_trace` | artifact | profile | str | path | absent unless trace capture requested | diagnostic | environment |
| `profile:timing_measured` | table | timing_measured | bool | 1 | never | diagnostic | environment |
| `profile:kernel_us/*` | table | kernel_us | float | us per iteration | event absent | diagnostic | environment |
| `result:bench/available` | table | bench | bool | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/unavailable_reason` | table | bench | str | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/device_index` | table | bench | int | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/device_name` | table | bench | str | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/device_capability` | table | bench | str | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/device_uuid` | table | bench | str | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/warmup_iters` | table | bench | int | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/rounds` | table | bench | int | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/iters_per_round` | table | bench | int | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/seed` | table | bench | int | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/order` | table | bench | str | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/graph_all_capturable` | table | bench | bool | 1 | disabled/unsupported; UUID may be unavailable | diagnostic | environment |
| `result:bench/eager/event_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/event_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/event_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/event_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/event_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/event_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/event_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/wall_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/wall_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/wall_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/wall_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/wall_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/wall_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/wall_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/graph_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/graph_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/graph_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/graph_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/graph_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/graph_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/graph_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/eager/prepare_s` | table | bench | float | s | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/eager/warmup_iters` | table | bench | int | iterations | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/eager/graph_capturable` | table | bench | bool | 1 | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/eager/graph_error` | table | bench | str | text | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/eager/graph_replay_ok` | table | bench | bool | 1 | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `benchmark_raw:samples/eager/event_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/eager/wall_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/eager/graph_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/eager/prepare_s` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/eager/warmup_iters` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/eager/graph_capturable` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/eager/graph_error` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/eager/graph_replay_ok` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `result:bench/compile/event_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/event_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/event_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/event_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/event_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/event_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/event_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/wall_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/wall_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/wall_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/wall_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/wall_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/wall_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/wall_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/graph_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/graph_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/graph_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/graph_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/graph_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/graph_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/graph_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/compile/prepare_s` | table | bench | float | s | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/compile/warmup_iters` | table | bench | int | iterations | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/compile/graph_capturable` | table | bench | bool | 1 | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/compile/graph_error` | table | bench | str | text | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/compile/graph_replay_ok` | table | bench | bool | 1 | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `benchmark_raw:samples/compile/event_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/compile/wall_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/compile/graph_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/compile/prepare_s` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/compile/warmup_iters` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/compile/graph_capturable` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/compile/graph_error` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/compile/graph_replay_ok` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `result:bench/generated/event_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/event_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/event_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/event_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/event_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/event_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/event_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/wall_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/wall_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/wall_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/wall_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/wall_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/wall_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/wall_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/graph_us_n` | table | bench | int | samples | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/graph_us_median` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/graph_us_min` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/graph_us_max` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/graph_us_p25` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/graph_us_p75` | table | bench | float | us | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/graph_cv` | table | bench | float | ratio | disabled/unavailable/graph not captured | diagnostic | environment |
| `result:bench/generated/prepare_s` | table | bench | float | s | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/generated/warmup_iters` | table | bench | int | iterations | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/generated/graph_capturable` | table | bench | bool | 1 | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/generated/graph_error` | table | bench | str | text | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `result:bench/generated/graph_replay_ok` | table | bench | bool | 1 | disabled/unavailable; graph_error null on success; replay null on capture failure | diagnostic | environment |
| `benchmark_raw:samples/generated/event_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/generated/wall_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/generated/graph_us` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/generated/prepare_s` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/generated/warmup_iters` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/generated/graph_capturable` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/generated/graph_error` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `benchmark_raw:samples/generated/graph_replay_ok` | artifact | samples | list/number/bool/str | us for *_us; s for prepare_s; otherwise 1 | not sampled/graph off | diagnostic | environment |
| `result:bench/generated/event_wall_ratio` | table | bench | float | ratio | disabled/unavailable/zero wall time | diagnostic | environment |
| `benchmark_raw:order` | artifact | order | list[str] | 1 | never | diagnostic | environment |
| `batch:adv/frac_dropped` | metric | adv | float | 1 | no complete batch | diagnostic | environment |
| `batch:adv/mean` | metric | adv | float | 1 | no complete batch | diagnostic | environment |
| `batch:adv/std` | metric | adv | float | 1 | no complete batch | diagnostic | environment |
| `batch:adv/gpu_busy_excluded` | metric | adv | int | rollouts | no complete batch | excluded environment failure count | environment |
| `batch:reward/mean` | metric | reward | float | reward | no complete batch | reward aggregate | environment |
| `batch:reward/frac_3` | metric | reward | float | 1 | no complete batch | reward aggregate | environment |
| `batch:reward/frac_provenance_fail` | metric | reward | float | 1 | no complete batch | reward aggregate | environment |
| `batch:reward/frac_max_turns` | metric | reward | float | 1 | no complete batch | reward aggregate | environment |
| `batch:reward/mean_overlong_penalty` | metric | reward | float | reward | no complete batch | reward aggregate | environment |
| `batch:turns/mean` | metric | turns | float | turns | no complete batch | diagnostic | environment |
| `batch:turns/p95` | metric | turns | float | turns | no complete batch | diagnostic | environment |
| `runtime:runtime/python` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/python_ok` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/gpu_count` | config | runtime | int | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/gpu0_name` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/gpu0_compute_cap` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/gpu0_uuid` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/driver_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/driver_ok` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/gpu0_cutile_supported` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/nvcc_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/nvcc_ok` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/ninja` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/tileiras_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/tileiras_ok` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/tileiras_reason` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/cuda_tile_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/torch_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/torch_cuda_build` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/triton_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/tilelang_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/cutlass_dsl_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/cuda_bindings_version` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/nsys` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/ncu` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/compute_sanitizer` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/ok_cuda_backend` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/ok_triton_backend` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/ok_cutile` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/ok_tilelang` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/ok_cutedsl` | config | runtime | bool | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `runtime:runtime/unavailable_reason` | config | runtime | str | 1 | tool/package/device unavailable; reason null on success | diagnostic | environment |
| `names:backends/*/profile/timing_measured` | table | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/timing_reason` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/t_eager_us` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/t_compile_us` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/kernel_us/*` | table | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/profile/*_events` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/profile/eager_kernel_names` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/profile/*_kernel_samples` | artifact | profile | str (JSON list of [name, calls, total_us]; per-event list in the kernel_samples_path sidecar since 2026-11-09) | us per CUDA event; non-finite durations encoded as strings | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/profile/kernel_samples_path` | artifact | profile | str | path to the per-event JSON sidecar | not written (no samples dir or OSError); absent on CPU and before 2026-11-09 | diagnostic | environment |
| `names:backends/*/profile/profile/order` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/profile/timing_method` | artifact | profile | str | versioned capture conditioning | absent historically/CPU | diagnostic | environment |
| `names:backends/*/profile/profile/primer_event_counts` | artifact | profile | str (JSON list) | CUDA events per conditioning capture | absent historically/CPU | diagnostic | environment |
| `names:backends/*/profile/profile/diagnostic` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/profile/profile/*_trace` | artifact | profile | scalar/string | us for timing; otherwise 1 | CPU/failure/disabled | diagnostic | environment |
| `names:backends/*/measured` | table | backends | bool | 1 | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/reason` | table | backends | str | 1 | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/unavailable` | table | backends | str | 1 | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/agent_kernel_names` | table | backends | list[str] | 1 | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/unmatched_agent_names` | table | backends | list[str] | 1 | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/stderr_tail` | weave | backends | str | 1 | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/t_gen_us` | table | backends | float | us | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/provenance` | table | backends | float | fraction | unavailable/unmeasured; reasons absent on success | diagnostic | environment |
| `names:backends/*/provenance/agent` | table | backends | float | fraction | unmeasured | diagnostic | environment |
| `names:backends/*/provenance/library` | table | backends | float | fraction | unmeasured | diagnostic | environment |
| `names:backends/*/provenance/torch` | table | backends | float | fraction | unmeasured | diagnostic | environment |
| `names:backends/*/provenance/memory` | table | backends | float | fraction | unmeasured | diagnostic | environment |
| `names:backends/*/kernels/*/us` | table | backends | float | us | no event | diagnostic | environment |
| `names:backends/*/kernels/*/class` | table | backends | str | category | no event | diagnostic | environment |
| `results:command` | config | command | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:device` | config | device | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:config_scope` | config | config_scope | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:provenance_rules` | config | provenance_rules | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:config/backend` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/backend` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `results:config/iters` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/iters` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `results:config/benchmark` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/benchmark` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `results:config/bench_warmup_iters` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/bench_warmup_iters` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `results:config/bench_rounds` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/bench_rounds` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `results:config/bench_iters_per_round` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/bench_iters_per_round` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `results:config/cuda_graphs` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `results:config/diagnostics` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `results:config/exact_integer_outputs` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `results:config/relative_error_bounds` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `results:config/gpu_lock` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `results:config/gpu_lock_timeout_s` | config | config | float | s | historical artifact did not persist it | config | environment |
| `results:case_configs/*/cuda_graphs` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/diagnostics` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/exact_integer_outputs` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/relative_error_bounds` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/gpu_lock` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/gpu_lock_timeout_s` | table | case_configs | float | s | historical artifact did not persist it | config | environment |
| `results:config/timeout_s` | config | config | float | s | historical artifact did not persist it | config | environment |
| `results:case_configs/*/timeout_s` | table | case_configs | float | s | historical artifact did not persist it | config | environment |
| `results:config/provenance_threshold` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/provenance_threshold` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `results:config/speedup_margin` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/speedup_margin` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `results:config/max_turns` | config | config | int | turns | historical artifact did not persist it | config | environment |
| `results:case_configs/*/max_turns` | table | case_configs | int | turns | historical artifact did not persist it | config | environment |
| `results:config/max_context_tokens` | config | config | int | tokens | historical artifact did not persist it | config | environment |
| `results:case_configs/*/max_context_tokens` | table | case_configs | int | tokens | historical artifact did not persist it | config | environment |
| `results:config/algo` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/algo` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `results:config/group_size` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/group_size` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `results:config/adv_norm` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/adv_norm` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `results:config/filter_tied_groups` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/filter_tied_groups` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `results:config/overlong_buffer_turns` | config | config | int | turns | historical artifact did not persist it | config | environment |
| `results:case_configs/*/overlong_buffer_turns` | table | case_configs | int | turns | historical artifact did not persist it | config | environment |
| `results:config/clip_low` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/clip_low` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `results:config/clip_high` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/clip_high` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `results:config/loss_agg` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/loss_agg` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `results:config/kl_coef` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/kl_coef` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `results:config/seed` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/seed` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `results:config/run_name` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/run_name` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `results:config/tags` | config | config | list[str] | 1 | historical artifact did not persist it | config | environment |
| `results:case_configs/*/tags` | table | case_configs | list[str] | 1 | historical artifact did not persist it | config | environment |
| `names:command` | config | command | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `names:device` | config | device | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `names:config_scope` | config | config_scope | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `names:provenance_rules` | config | provenance_rules | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `names:config/backend` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/backend` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `names:config/iters` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/iters` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `names:config/benchmark` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/benchmark` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `names:config/bench_warmup_iters` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/bench_warmup_iters` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `names:config/bench_rounds` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/bench_rounds` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `names:config/bench_iters_per_round` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/bench_iters_per_round` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `names:config/cuda_graphs` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `names:config/diagnostics` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `names:config/exact_integer_outputs` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `names:config/relative_error_bounds` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `names:config/gpu_lock` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `names:config/gpu_lock_timeout_s` | config | config | float | s | historical artifact did not persist it | config | environment |
| `names:case_configs/*/cuda_graphs` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/diagnostics` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/exact_integer_outputs` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/relative_error_bounds` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/gpu_lock` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/gpu_lock_timeout_s` | table | case_configs | float | s | historical artifact did not persist it | config | environment |
| `names:config/timeout_s` | config | config | float | s | historical artifact did not persist it | config | environment |
| `names:case_configs/*/timeout_s` | table | case_configs | float | s | historical artifact did not persist it | config | environment |
| `names:config/provenance_threshold` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/provenance_threshold` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `names:config/speedup_margin` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/speedup_margin` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `names:config/max_turns` | config | config | int | turns | historical artifact did not persist it | config | environment |
| `names:case_configs/*/max_turns` | table | case_configs | int | turns | historical artifact did not persist it | config | environment |
| `names:config/max_context_tokens` | config | config | int | tokens | historical artifact did not persist it | config | environment |
| `names:case_configs/*/max_context_tokens` | table | case_configs | int | tokens | historical artifact did not persist it | config | environment |
| `names:config/algo` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/algo` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `names:config/group_size` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/group_size` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `names:config/adv_norm` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/adv_norm` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `names:config/filter_tied_groups` | config | config | bool | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/filter_tied_groups` | table | case_configs | bool | 1 | historical artifact did not persist it | config | environment |
| `names:config/overlong_buffer_turns` | config | config | int | turns | historical artifact did not persist it | config | environment |
| `names:case_configs/*/overlong_buffer_turns` | table | case_configs | int | turns | historical artifact did not persist it | config | environment |
| `names:config/clip_low` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/clip_low` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `names:config/clip_high` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/clip_high` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `names:config/loss_agg` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/loss_agg` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `names:config/kl_coef` | config | config | float | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/kl_coef` | table | case_configs | float | 1 | historical artifact did not persist it | config | environment |
| `names:config/seed` | config | config | int | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/seed` | table | case_configs | int | 1 | historical artifact did not persist it | config | environment |
| `names:config/run_name` | config | config | str | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/run_name` | table | case_configs | str | 1 | historical artifact did not persist it | config | environment |
| `names:config/tags` | config | config | list[str] | 1 | historical artifact did not persist it | config | environment |
| `names:case_configs/*/tags` | table | case_configs | list[str] | 1 | historical artifact did not persist it | config | environment |
| `results:group_configs/*/backend` | config | group_configs | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/iters` | config | group_configs | int | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/benchmark` | config | group_configs | bool | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/bench_warmup_iters` | config | group_configs | int | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/bench_rounds` | config | group_configs | int | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/bench_iters_per_round` | config | group_configs | int | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/cuda_graphs` | config | group_configs | bool | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/diagnostics` | config | group_configs | bool | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/exact_integer_outputs` | config | group_configs | bool | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/relative_error_bounds` | config | group_configs | bool | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/gpu_lock` | config | group_configs | bool | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/gpu_lock_timeout_s` | config | group_configs | float | s | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/timeout_s` | config | group_configs | float | s | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/provenance_threshold` | config | group_configs | float | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/speedup_margin` | config | group_configs | float | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/max_turns` | config | group_configs | int | turns | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/max_context_tokens` | config | group_configs | int | tokens | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/algo` | config | group_configs | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/group_size` | config | group_configs | int | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/adv_norm` | config | group_configs | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/filter_tied_groups` | config | group_configs | bool | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/overlong_buffer_turns` | config | group_configs | int | turns | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/clip_low` | config | group_configs | float | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/clip_high` | config | group_configs | float | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/loss_agg` | config | group_configs | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/kl_coef` | config | group_configs | float | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/seed` | config | group_configs | int | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/run_name` | config | group_configs | str | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:group_configs/*/tags` | config | group_configs | list[str] | 1 | historical artifact did not persist it | diagnostic | environment |
| `results:versions/*` | config | versions | str | version | package unavailable | diagnostic | environment |
| `results:candidate_sha256/*` | table | candidate_sha256 | str | sha256 | historical artifact missing | diagnostic | environment |
| `results:unavailable_backends/*` | summary | unavailable_backends | str | reason | none unavailable | diagnostic | environment |
| `results:measured_groups/*/advantages/*` | table | measured_groups | float | normalized reward | group unavailable | diagnostic | environment |
| `results:synthetic_groups_not_gpu_measurements/*` | artifact | synthetic_groups_not_gpu_measurements | JSON | synthetic | none generated | diagnostic | environment |
| `results:measured_groups/*/stats/adv/frac_dropped` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/adv/mean` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/adv/std` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/adv/gpu_busy_excluded` | metric | measured_groups | int | rollouts | group unavailable | excluded environment failure count | environment |
| `results:measured_groups/*/stats/reward/mean` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/reward/frac_3` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/reward/frac_provenance_fail` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/reward/frac_max_turns` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/reward/mean_overlong_penalty` | metric | measured_groups | float | 1 | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/turns/mean` | metric | measured_groups | float | turns | group unavailable | diagnostic | environment |
| `results:measured_groups/*/stats/turns/p95` | metric | measured_groups | float | turns | group unavailable | diagnostic | environment |
| `summary:cases` | summary | cases | int | count | never | diagnostic | environment |
| `summary:verify_ok` | summary | verify_ok | int | count | never | diagnostic | environment |
| `summary:provenance_pass` | summary | provenance_pass | int | count | never | diagnostic | environment |
| `summary:reward_positive` | summary | reward_positive | int | count | never | diagnostic | environment |
| `summary:reward_mean` | summary | reward_mean | float | reward | no cases | diagnostic | environment |
| `summary:unmatched_agent_names_total` | summary | unmatched_agent_names_total | int | count | never | diagnostic | environment |
| `summary:backends_measured` | summary | backends_measured | int | count | never | diagnostic | environment |
| `summary:gpu_tests` | summary | gpu_tests | str | status | test log missing | diagnostic | environment |
| `summary:unavailable_backends` | summary | unavailable_backends | str | status | never | diagnostic | environment |
| `row:meta/model_new_sha256` | table | meta | str | sha256 | not recorded historically | diagnostic | environment |
| `row:meta/provenance_rules` | table | meta | str | 1 | not recorded historically | diagnostic | environment |
| `row:case` | table | case | str | case identifier | never | diagnostic | environment |
| `row:case_config` | table | case_config | str | JSON config | historical per-case config missing | diagnostic | environment |
| `row:timing_measured` | table | timing_measured | bool | 1 | raw profile missing | diagnostic | environment |
| `row:profile/timing_method` | table | profile | str | versioned capture conditioning | raw profile missing/historical | diagnostic | environment |
| `tracking_config:meta/provenance_rules` | config | meta | str | identifier | historical classifier identity missing | diagnostic | environment |
| `tracking_config:source_command` | config | source_command | str | command | source command missing | diagnostic | environment |
| `tracking_config:tracking/contract` | config | tracking | str | identifier | never | diagnostic | environment |
| `tracking_config:tracking/config_available` | config | tracking | bool | 1 | never; false when source metadata missing | diagnostic | environment |
| `tracking_config:tracking/rules_available` | config | tracking | bool | 1 | never; false when source metadata missing | diagnostic | environment |
| `tracking_config:tracking/shares_available` | config | tracking | bool | 1 | never; false when source metadata missing | diagnostic | environment |
| `tracking_config:tracking/candidates_available` | config | tracking | bool | 1 | never; false when source metadata missing | diagnostic | environment |
| `tracking_config:tracking/group_configs` | config | tracking | dict | 1 | historical grouping config missing | diagnostic | environment |
| `tracking_config:versions/*` | config | versions | str | version | package unavailable | diagnostic | environment |
| `metric:iteration` | metric | iteration | int | iteration/snapshot | caller must provide training iteration | diagnostic | environment |
| `artifact_header:run_id` | artifact | run_id | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:files` | artifact | files | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:status` | artifact | status | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:kind` | artifact | kind | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:notes` | artifact | notes | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:tags` | artifact | tags | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:source_sha256/*` | artifact | source_sha256 | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:schema_version` | artifact | schema_version | JSON | 1 | not provided | diagnostic | environment |
| `artifact_header:timestamps/*` | artifact | timestamps | JSON | 1 | not provided | diagnostic | environment |
| `weave:workdir` | weave | workdir | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:task_id` | weave | task_id | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:backend` | weave | backend | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:cfg/*` | weave | cfg | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:turns_used` | weave | turns_used | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:model_new_sha256` | weave | model_new_sha256 | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:model_new_source` | weave | model_new_source | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:kernel_files` | weave | kernel_files | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:result/*` | weave | result | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:profile/*` | weave | profile | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:backfilled` | weave | backfilled | JSON | field-specific | input unavailable | diagnostic | environment |
| `weave:source` | weave | source | JSON | field-specific | input unavailable | diagnostic | environment |
| `trainer:train/entropy` | metric | train | float | nats/policy token | zero policy tokens, incomplete actor/token data, or non-finite entropy | trainer-owned diagnostic | trainer |
| `trainer:train/is_ratio_clip_fraction` | metric | train | float | fraction of policy tokens | zero policy tokens, missing/invalid log-probabilities or clip bounds | trainer-owned diagnostic | trainer |
| `trainer:train/trajectory_length_tokens` | metric | train | float | tokens/episode | empty collection or any unknown token ledger | trainer-owned diagnostic | trainer |
| `trainer:train/trajectory_length_turns` | metric | train | float | agent turns/episode | empty collection or unknown/inconsistent turn count | trainer-owned diagnostic | trainer |
| `trainer:eval/pass_rate` | metric | eval | float | weighted fraction of attempted episodes | empty/incomplete cohort or unknown weights/protocol | trainer-owned diagnostic | trainer |
| `trainer:eval/faster_rate_vs_compile` | metric | eval | float | weighted fraction of attempted episodes | empty/incomplete cohort or unknown weights/protocol | trainer-owned diagnostic | trainer |
| `trainer:eval/geomean_speedup_vs_compile` | metric | eval | float | ratio | empty/incomplete cohort, zero accepted weight, or invalid/underflowed aggregate | trainer-owned diagnostic | trainer |
| `trainer:train/collapse_step` | summary | train | int | iteration (zero-based) | detector unavailable or no confirmed collapse | trainer-owned diagnostic | trainer |
| `metric:adv/frac_dropped` | metric | adv | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/adv/frac_dropped` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:adv/mean` | metric | adv | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/adv/mean` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:adv/std` | metric | adv | float | 1 | no batch | diagnostic | environment |
| `metric:adv/gpu_busy_excluded` | metric | adv | int | rollouts | no batch | excluded environment failure count | environment |
| `metric:batch/*/adv/std` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/adv/gpu_busy_excluded` | metric | batch | int | rollouts | no batch | excluded environment failure count | environment |
| `metric:reward/mean` | metric | reward | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/reward/mean` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:reward/frac_3` | metric | reward | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/reward/frac_3` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:reward/frac_provenance_fail` | metric | reward | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/reward/frac_provenance_fail` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:reward/frac_max_turns` | metric | reward | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/reward/frac_max_turns` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:reward/mean_overlong_penalty` | metric | reward | float | 1 | no batch | diagnostic | environment |
| `metric:batch/*/reward/mean_overlong_penalty` | metric | batch | float | 1 | no batch | diagnostic | environment |
| `metric:turns/mean` | metric | turns | float | turns | no batch | diagnostic | environment |
| `metric:batch/*/turns/mean` | metric | batch | float | turns | no batch | diagnostic | environment |
| `metric:turns/p95` | metric | turns | float | turns | no batch | diagnostic | environment |
| `metric:batch/*/turns/p95` | metric | batch | float | turns | no batch | diagnostic | environment |
| `metric:rollouts` | table | rollouts | Table | rows | table unavailable | diagnostic | environment |
| `metric:kernels` | table | kernels | Table | rows | table unavailable | diagnostic | environment |
| `kernel_row:backend` | table | backend | JSON scalar | backend | not supplied | diagnostic | environment |
| `kernel_row:event` | table | event | JSON scalar | symbol | not supplied | diagnostic | environment |
| `kernel_row:us` | table | us | JSON scalar | us/iteration | not supplied | diagnostic | environment |
| `kernel_row:class` | table | class | JSON scalar | category | not supplied | diagnostic | environment |
| `kernel_row:measured` | table | measured | JSON scalar | bool | not supplied | diagnostic | environment |
| `kernel_row:reason` | table | reason | JSON scalar | text | not supplied | diagnostic | environment |
| `kernel_row:unmatched` | table | unmatched | JSON scalar | authored names | not supplied | diagnostic | environment |
| `result:rollout_id` | table | rollout_id | str/int | identifier | caller omitted | diagnostic | environment |
| `stage:cmd` | artifact | cmd | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `stage:returncode` | artifact | returncode | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `stage:timed_out` | artifact | timed_out | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `stage:stdout` | artifact | stdout | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `stage:stderr` | artifact | stderr | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `stage:stdout_tail` | artifact | stdout_tail | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `stage:stderr_tail` | artifact | stderr_tail | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `stage:nvidia_smi` | artifact | nvidia_smi | JSON scalar | sensitive raw diagnostic | tool unavailable | diagnostic | environment |
| `artifact_header:gpu` | config | gpu | str | requested GPU class (not measured device) | not supplied | diagnostic | environment |
| `artifact_header:_comment` | drop | _comment | str | explanatory text | not supplied | diagnostic | environment |

### Progressive-task-only projections

These keys exist only in the opt-in task-suite API. They do not alter the stock
rollout schema or classifier. `task/provenance_authored` excludes memory and
library credit; the approved new-task gate prohibits library time. The legacy
combined share and reward are retained for auditing. Warm/cold benchmark and
ceiling fields never enter reward computation. The 60%-cuBLAS fixture acceptance
check is separate from milestone rewards. Cold means the first post-initialization
forward in a benchmark process, not flushed hardware caches.

| Source:key | Category | Namespace | Type | Unit | Null/absence condition | Role | Owner |
|---|---|---|---|---|---|---|---|
| `result:task/provenance_policy` | table | task | str | policy id | not a suite row | identity | task suite |
| `result:task/provenance_authored` | table | task | float | fraction of non-memory time | missing/invalid capture | task-only reward/input | task suite |
| `result:task/provenance_threshold` | table | task | float | fraction | not a suite row | task-only reward/config | task suite |
| `result:task/legacy_reward_value` | table | task | float | milestone | raw evaluator unavailable | audit diagnostic | task suite |
| `result:task/policy_violation` | table | task | str | reason | no policy violation | diagnostic | task suite |
| `result:task/performance_required` | table | task | bool | 1 | not evaluated | fixture acceptance | task suite |
| `result:task/minimum_cublas_fraction` | table | task | float | ratio | no required performance gate | fixture acceptance/config | task suite |
| `result:task/cublas_fraction` | table | task | float | ratio | ceiling not measured | diagnostic | task suite |
| `result:task/performance_accepted` | table | task | bool | 1 | ceiling not measured/not required | fixture acceptance | task suite |
| `result:suite/task` | table | suite | str | task id | never for suite rows | identity | task suite |
| `result:suite/case` | table | suite | str | case id | never for suite rows | identity | task suite |
| `result:suite/stage` | table | suite | int | curriculum stage | never for suite rows | identity | task suite |
| `result:suite/backend` | table | suite | str | backend id | never for suite rows | identity | task suite |
| `result:suite/fixture` | table | suite | str | fixture kind | never for suite rows | identity | task suite |
| `result:suite/purpose` | table | suite | str | eval/train | never for suite rows | licensing scope | task suite |
| `result:suite/eval_only` | table | suite | bool | 1 | never for suite rows | licensing scope | task suite |
| `result:suite/contract_only` | table | suite | bool | 1 | never for suite rows | coverage scope | task suite |
| `result:suite/status` | table | suite | str | passed/failed/unavailable | never for suite rows | coverage | task suite |
| `result:suite/reason` | table | suite | str | reason | no reason | diagnostic | task suite |
| `result:suite/wall_s` | table | suite | float | s | case not executed | deadline diagnostic | task suite |
| `result:suite/numerics_unavailable_reason` | table | suite | str | reason | diagnostic succeeded/not requested | diagnostic only; never reward | task suite |
| `result:bench/cold_definition` | table | bench | str | timing definition | cold sampling disabled | diagnostic | environment |
| `result:bench/*/cold_event_us` | table | bench | float | us | cold sampling disabled/unavailable | diagnostic | environment |
| `result:bench/*/cold_wall_us` | table | bench | float | us | cold sampling disabled/unavailable | diagnostic | environment |
| `result:bench/workload/flops` | table | bench | int | algorithmic FLOP | count undefined | diagnostic | task suite |
| `result:bench/workload/logical_io_bytes` | table | bench | int | logical bytes | count undefined | diagnostic | task suite |
| `result:bench/workload/flop_convention` | table | bench | str | count definition | not evaluated | diagnostic | task suite |
| `result:bench/workload/bandwidth_convention` | table | bench | str | count definition | not evaluated | diagnostic | task suite |
| `result:bench/generated/tflops` | table | bench | float | TFLOP/s | warm timing/count unavailable | diagnostic | task suite |
| `result:bench/generated/logical_gbytes_per_s` | table | bench | float | logical GB/s | warm timing/count unavailable | diagnostic | task suite |
| `result:ceiling/available` | table | ceiling | bool | 1 | benchmark disabled | diagnostic | task suite |
| `result:ceiling/unavailable_reason` | table | ceiling | str | reason | ceiling available | diagnostic | task suite |
| `result:ceiling/provider` | table | ceiling | str | library id | unavailable | diagnostic | task suite |
| `result:ceiling/definition` | table | ceiling | str | timing definition | unavailable | diagnostic | task suite |
| `result:ceiling/warmup_iters` | table | ceiling | int | iterations | unavailable | diagnostic | task suite |
| `result:ceiling/rounds` | table | ceiling | int | rounds | unavailable | diagnostic | task suite |
| `result:ceiling/iters_per_round` | table | ceiling | int | iterations | unavailable | diagnostic | task suite |
| `result:ceiling/seed` | table | ceiling | int | seed | unavailable | diagnostic | task suite |
| `result:ceiling/dtype` | table | ceiling | str | dtype | unavailable | diagnostic | task suite |
| `result:ceiling/m` | table | ceiling | int | rows | unavailable | diagnostic | task suite |
| `result:ceiling/n` | table | ceiling | int | columns | unavailable | diagnostic | task suite |
| `result:ceiling/k` | table | ceiling | int | reduction elements | unavailable | diagnostic | task suite |
| `result:ceiling/flops` | table | ceiling | int | FLOP | unavailable | diagnostic | task suite |
| `result:ceiling/order` | table | ceiling | str | round order | unavailable | diagnostic | task suite |
| `result:ceiling/*_event_us_median` | table | ceiling | float | us | unavailable | diagnostic | task suite |
| `result:ceiling/*_tflops` | table | ceiling | float | TFLOP/s | unavailable | diagnostic | task suite |
| `result:ceiling/cublas_fraction` | table | ceiling | float | ratio | unavailable | diagnostic | task suite |
| `result:ceiling/device_name` | table | ceiling | str | measured device | unavailable | diagnostic | task suite |
| `result:ceiling/device_uuid` | table | ceiling | str | device id | unavailable | diagnostic | task suite |
| `result:ceiling/device_capability` | table | ceiling | str | SM version | unavailable | diagnostic | task suite |
| `result:ceiling/torch_version` | table | ceiling | str | version | unavailable | diagnostic | task suite |
| `result:ceiling/cuda_version` | table | ceiling | str | version | unavailable | diagnostic | task suite |
| `result:ceiling/nvcc_version` | table | ceiling | str | version output | unavailable | diagnostic | task suite |

### Task-suite W&B runs (`tracking/suite_log.py`)

One W&B run per downloaded task-suite evidence folder (run id
`h200-<suite run_id>`, job type `task-suite-h200`). Suite rows are evaluation
records, not training metrics: they live in the `suite_rows` table, and the
row keys keep their `result:` classification above (`log/*` and `artifact/*`
are dropped). Counts are logged once at iteration 0.

| Source:key | Category | Namespace | Type | Unit | Null/absence condition | Role | Owner |
|---|---|---|---|---|---|---|---|
| `suite_wandb:suite/request` | config | suite | JSON object | request | never | identity | task suite |
| `suite_wandb:suite/case_set` | config | suite | str | case selection | never | identity | task suite |
| `suite_wandb:suite/backends` | config | suite | str | backend list | never | identity | task suite |
| `suite_wandb:suite/fixtures` | config | suite | str | fixture list | never | identity | task suite |
| `suite_wandb:suite/tasks` | config | suite | list[str] | task ids | never | identity | task suite |
| `suite_wandb:suite/gpu_requested` | config | suite | str | requested GPU class | not recorded | diagnostic | task suite |
| `suite_wandb:source/manifest_sha256` | config | source | str | SHA-256 | no manifest downloaded | provenance | task suite |
| `suite_wandb:source/git_revision` | config | source | str | git commit | no manifest downloaded | provenance | task suite |
| `suite_wandb:source/files` | config | source | int | files | no manifest downloaded | provenance | task suite |
| `suite_wandb:suite/passed` | summary | suite | int | rows | never | coverage | task suite |
| `suite_wandb:suite/failed` | summary | suite | int | rows | never | coverage | task suite |
| `suite_wandb:suite/unavailable` | summary | suite | int | rows | never | coverage | task suite |
| `suite_wandb:suite/status` | summary | suite | str | pass/partial/fail | never | coverage | task suite |
| `suite_wandb:suite/source_mismatches` | summary | suite | int | files | never | provenance | task suite |
| `suite_wandb:suite/wall_s` | summary | suite | float | s | execution record missing | deadline diagnostic | task suite |
| `suite_wandb:suite_rows` | table | suite | wandb.Table | rows | never | evaluation record | task suite |

