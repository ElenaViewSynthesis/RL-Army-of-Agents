# TrajectoryOS — Architecture Note

## What this is

A modular platform for supervised fine-tuning, GRPO/GSPO reinforcement learning, long-context
multi-turn agent training, synthetic-data generation, verifiable environments,
planner–executor–verifier agent architectures, model/tool routing, continual improvement and
cost-aware deployment — built **around** THUDM/slime rather than on a fork of it.

## The one invariant that shapes everything

> Trainable trajectories preserve the **exact token IDs sampled by the rollout model**.

Decoding model output to text and re-tokenizing it produces token sequences that can differ
from what the policy actually sampled (merge ambiguity, special tokens, chat-template drift),
which corrupts importance ratios and log-prob alignment in RL. Therefore:

- `TrajectoryEvent.token_ids` are captured at generation time from the inference engine
  (SGLang `return_logprob=True` output), together with `rollout_logprobs`.
- `loss_mask` is constructed structurally, not by string matching:
  - policy-sampled tokens chosen for optimisation → `1`
  - tool observations, environment responses, chat-template scaffolding, system and user
    messages → `0`
- These invariants are **enforced by schema validators**: a non-assistant event with any
  `loss_mask=1` token is rejected at construction time; `token_ids`, `loss_mask` and
  `rollout_logprobs` must be length-aligned.

## slime as the RL substrate

slime owns training (Megatron/FSDP backends), weight sync, and rollout scheduling. We plug in
through its documented extension points and nothing else:

| slime flag | TrajectoryOS component |
|---|---|
| `--custom-generate-function-path` | `integrations/slime/generate.py` — runs the agent loop against SGLang, returns `Sample`s with exact `tokens`, `loss_mask`, `rollout_log_probs` |
| `--custom-rm-path` | `integrations/slime/reward.py` — wraps `trajectoryos.rewards.compute_reward` |
| `--data-source-path` | `integrations/slime/data_source.py` — streams `TaskSpec`s from datasets |
| `--rollout-function-path` | only if global rollout orchestration is required (e.g. cross-sample curriculum); default is to keep slime's rollout loop |
| `--eval-function-path` | `integrations/slime/eval.py` — baseline-vs-checkpoint evaluation |
| `--custom-rollout-log-function-path` | trajectory + reward-component logging into the trajectory store |
| `--custom-loss-function-path` | experimental objectives only (never needed for stock GRPO/GSPO) |

GRPO ↔ GSPO is a slime-side configuration change; TrajectoryOS keeps every non-algorithm
variable fixed so the comparison is controlled. The model is referenced only through
`ModelSpec` (`configs/models/*.yaml`) — swapping the dense Qwen3-4B-Base for a Qwen3 MoE
checkpoint for GSPO requires a config change, not a code change.

## Repository layout

```
advancedRL/
  apps/
    api/                 # FastAPI backend (Milestone 10)
    web/                 # Next.js dashboard (Milestone 10)
  packages/              # uv workspace members, all under the trajectoryos.* namespace
    core/                # config loading, structured logging, seeding
    schemas/             # canonical Pydantic models (TaskSpec, Trajectory, ...)
    trajectory_store/    # Parquet + object-store persistence (M3+)
    environments/        # sandbox adapters; Docker repo sandbox first (M2)
    verifiers/           # clean-sandbox verifiers (M2)
    synthetic_data/      # generation → verification → curriculum pipeline (M6)
    rewards/             # composable cost-aware reward
    agents/              # agent loops, tools, planner–executor–verifier (M2+)
    model_router/        # budget-conditioned model/tool routing (M8)
    evaluation/          # eval harnesses, Pareto analysis (M4+)
    deployment/          # serving + rollback (M9/M10)
  integrations/
    slime/               # the only place slime interfaces are touched
    sglang/              # inference client preserving token IDs + logprobs
    huggingface/         # model/dataset hub adapters
    hermes_agent/        # optional external agent-harness adapter
  configs/
    models/  environments/  rewards/  training/  evaluation/
  tests/
    unit/  integration/  trajectory_correctness/  reward_correctness/  smoke/
  docs/
```

Empty directories carry a `.gitkeep` and gain code only when their milestone lands — no
placeholder implementations.

## Canonical data model (`trajectoryos.schemas`)

- **TaskSpec** — `task_id`, `prompt`, `context`, `tools`, `environment` (`EnvironmentRef`),
  `verifier` (`VerifierRef`), `budget` (`BudgetSpec`), `metadata`.
- **BudgetSpec** — `max_input_tokens`, `max_output_tokens`, per-tool `max_tool_calls`,
  `max_wallclock_seconds`, `max_gpu_seconds`, `max_estimated_cost_usd`. `None` means
  unbounded; enforcement happens in the agent loop (M2).
- **TrajectoryEvent** — append-only record of everything that happens in a rollout:
  ids/parentage, `event_type`, `role`, `timestamp`, `content`, `token_ids`,
  `rollout_logprobs`, `loss_mask`, tool call/result fields, `latency_ms`, `estimated_cost`,
  `metadata`.
- **Trajectory** — `run_id`, `rollout_id`, `task_id`, `policy_version`, `events`,
  `terminal_state` (+ `termination_reason` — truncated/failed rollouts are recorded, never
  dropped), `reward_components`, `total_reward`, `verifier_result`, `cost_summary`.
  `Trajectory.token_stream()` flattens events into aligned `(token_ids, loss_mask,
  rollout_logprobs)` for conversion to slime `Sample`s.
- **RewardResult** — `task_success`, `quality`, `progress`, `format_compliance`,
  `token_cost`, `tool_cost`, `latency_cost`, `gpu_cost`, `retry_cost`, `safety_penalty`,
  `total_reward`.

## Cost-aware reward (`trajectoryos.rewards`)

```
reward =  success_weight  * task_success
        + quality_weight  * quality
        + progress_weight * progress
        + format_weight   * format_compliance
        - token_penalty   * normalized_token_cost
        - tool_penalty    * normalized_tool_cost
        - latency_penalty * normalized_latency
        - gpu_penalty     * normalized_gpu_seconds
        - retry_penalty   * retry_count
        - safety_weight   * safety_penalty
```

`normalize_costs` maps raw `CostSummary` usage against `BudgetSpec` caps (usage/cap, may
exceed 1.0 when over budget; 0.0 when a cap is unbounded). `reward_breakdown` returns each
signed weighted contribution so every component is independently logged and testable; the
breakdown provably sums to `total_reward` (tested).

## Configuration

YAML files under `configs/`, validated into Pydantic models via
`trajectoryos.core.load_config`. Environment overrides use double-underscore paths:
`TRAJECTORYOS__model__hf_repo=Qwen/Qwen3-30B-A3B-Base` overrides `model.hf_repo`. Override
values are YAML-parsed, so numbers, booleans and lists round-trip with correct types. All
schemas use `extra="forbid"` — a typo in a config key is an error, not a silent no-op.

## Quality gates

- `ruff format` + `ruff check` (lint), `mypy --strict` (all packages typed), `pytest`.
- Deterministic seeding via `trajectoryos.core.seed_everything` and `derive_seed` for stable
  per-task sub-seeds.
- Structured logging via structlog; reward components and termination reasons are always
  logged as fields, not prose.
- Real infrastructure boundaries: when GPUs/SGLang/slime are unavailable locally, tests use
  explicitly labelled mocks and the docs carry the exact command for the real environment.
