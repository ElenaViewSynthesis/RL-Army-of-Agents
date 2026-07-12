# TrajectoryOS — Implementation Plan

TrajectoryOS is an open-source post-training and agentic-RL platform built **around**
[THUDM/slime](https://github.com/THUDM/slime) as the RL substrate. We never fork or rewrite
slime; we integrate exclusively through its supported extension points
(`--custom-generate-function-path`, `--custom-rm-path`, `--data-source-path`,
`--rollout-function-path`, `--eval-function-path`, `--custom-rollout-log-function-path`,
`--custom-loss-function-path`).

**Core invariant (applies to every milestone):** trainable trajectories preserve the exact
token IDs sampled by the rollout model. Training targets are never recovered by decoding and
re-tokenizing text. Tool observations, environment responses, templates and user messages carry
`loss_mask=0`; policy-generated tokens selected for optimisation carry `loss_mask=1`.

## Milestone status

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Repository, schemas, config loading, unit tests | ✅ done |
| 2 | Local Docker sandbox environment + bug-fix task + verifier + budgets | ⬜ next |
| 3 | SGLang + slime adapter (custom generate fn, token/logprob/mask preservation, Sample conversion, custom RM, smoke test) | ⬜ |
| 4 | GRPO baseline (multi-candidate rollouts, eval, metric logging) | ⬜ |
| 5 | GSPO experiment (algorithm switch, fixed variables, MoE-ready model adapter) | ⬜ |
| 6 | Synthetic-data pipeline (Distilabel-style generation → verification → versioned Parquet/JSONL) | ⬜ |
| 7 | Long-context memory agent (READ / WRITE_MEMORY / UPDATE_MEMORY / RETRIEVE / COMPACT / ANSWER; 32K → 256K) | ⬜ |
| 8 | Model & tool orchestrator (budget-conditioned planner over frozen experts) | ⬜ |
| 9 | Continual-learning pipeline (quarantine buffer, replay, canary gates) | ⬜ |
| 10 | Web platform (FastAPI + Postgres + Parquet/MinIO + Next.js dashboard) | ⬜ |

## Milestone 1 — delivered

- `docs/ARCHITECTURE.md` — architecture note covering layout, slime integration boundaries,
  and the token/loss-mask invariant.
- `packages/schemas` (`trajectoryos.schemas`) — Pydantic v2 models: `TaskSpec`, `BudgetSpec`,
  `TrajectoryEvent`, `Trajectory`, `RewardResult`, plus supporting `ToolSpec`,
  `EnvironmentRef`, `VerifierRef`, `CostSummary`, `VerifierResult`, `TerminalState`,
  `ModelSpec`. Loss-mask invariants are enforced by validators at the event and trajectory
  level.
- `packages/core` (`trajectoryos.core`) — YAML config loading with deep-merge and
  `TRAJECTORYOS__section__key` environment overrides, structured logging (structlog),
  deterministic seeding helpers.
- `packages/rewards` (`trajectoryos.rewards`) — composable cost-aware reward:
  `compute_reward`, `reward_breakdown` (independently-logged signed contributions),
  `normalize_costs` (usage vs. `BudgetSpec` caps).
- `configs/` — `models/qwen3-4b-base.yaml` (model identifier configurable; MoE flag for later
  GSPO experiments), `rewards/default.yaml`, `training/grpo_baseline.yaml`.
- `tests/unit`, `tests/reward_correctness`, `tests/trajectory_correctness` — schema
  validation, config loading/overrides, reward composition, loss-mask correctness.
- Tooling: uv workspace, ruff (format + lint), mypy strict, pytest.

## Milestone 2 — plan (next)

1. `packages/environments`: `Environment` protocol + local Docker sandbox adapter
   (create container from image, mount a task repo copy, exec with timeouts, teardown).
2. Task type: repository bug-fixing (`configs/environments/repo_bugfix.yaml` + a tiny fixture
   repo with a failing unit test under `tests/integration/fixtures/`).
3. `packages/agents`: tools `read_file`, `edit_file`, `run_shell` — every invocation recorded
   as `TrajectoryEvent`s (`TOOL_CALL` + `TOOL_RESULT`, `loss_mask=0`).
4. `packages/verifiers`: clean-sandbox unit-test verifier — applies the agent's patch to a
   pristine checkout in a **second** container, restores the canonical test files, runs the
   test suite. The agent's container never touches the evaluation container (anti
   reward-hacking boundary).
5. Budget enforcement: token, per-tool call and wallclock budgets checked before each step;
   violations terminate with `TerminalState.BUDGET_EXCEEDED` and a recorded reason.
6. Integration tests for sandbox isolation (skipped automatically when Docker is absent) and
   unit tests with a mocked container runtime (explicitly labelled mocks).

## Later milestones — key decisions locked in early

- **M3**: the custom generate function calls SGLang with `return_logprob=True` and builds
  `Sample.tokens` / `Sample.loss_mask` / `Sample.rollout_log_probs` directly from returned
  token IDs — no re-tokenization. Multi-turn tool episodes append observation tokens with
  mask 0. Smoke test runs against mocked inference; a runnable command for a real GPU
  environment is documented.
- **M4/M5**: GRPO vs GSPO is a config switch (`training.algorithm`); everything else in the
  launch config is held fixed. Model adapter reads `ModelSpec.is_moe` so a Qwen3 MoE model
  drops in for GSPO without code changes.
- **M9**: production weights are never updated from unverified traffic — quarantine buffer →
  redaction → isolated re-run → offline eval → canary gate → promotion.

## Conventions

- Python ≥ 3.11, Pydantic v2, `extra="forbid"` on all schemas.
- All packages are uv-workspace members exposing the `trajectoryos.*` namespace.
- `uv run ruff format . && uv run ruff check . && uv run mypy packages tests && uv run pytest`
  must pass before a milestone is considered complete.
- Failed or truncated trajectories are first-class data: they are stored with
  `terminal_state` + `termination_reason`, never silently discarded.
