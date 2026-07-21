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
| 2 | Local Docker sandbox environment + bug-fix task + verifier + budgets | ✅ done |
| 3 | SGLang + slime adapter (custom generate fn, token/logprob/mask preservation, Sample conversion, custom RM, smoke test) | ✅ done |
| 4 | GRPO baseline (multi-candidate rollouts, eval, metric logging) | ⬜ next |
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

## Milestone 2 — delivered

- `packages/environments` (`trajectoryos.environments`): `Sandbox` protocol with
  path-traversal guards; `DockerSandbox` (docker CLI, `--network none`, memory/CPU caps,
  repo copied in — no host mounts) and `LocalProcessSandbox` (real execution in a temp dir;
  explicitly documented as non-isolated, for tests/dev only).
- `packages/agents` (`trajectoryos.agents`): `read_file` / `edit_file` / `run_shell` tools,
  `BudgetTracker` (token, per-tool, wallclock, GPU, cost caps → reason strings),
  `Policy` protocol + `ScriptedPolicy` (deterministic test policy; tool calls really
  execute), and `run_episode` — records every step as `TrajectoryEvent`s, enforces budgets
  before/after each step, accumulates the patch, never discards failures
  (`terminal_state` + `termination_reason` always set).
- `packages/verifiers` (`trajectoryos.verifiers`): `CleanSandboxTestVerifier` — fresh
  pristine sandbox, protected globs reject patches to tests/configs (rejections logged),
  canonical test command decides success. Reward-hacking paths covered by tests.
- Fixture task `tests/fixtures/buggy_calculator` + `configs/environments/repo_bugfix.yaml`.
- Tests: budget enforcement, sandbox/tool behaviour, end-to-end episode → clean verification,
  reward-hacking rejection; Docker isolation integration tests auto-skip without a daemon
  (run with Docker up: `uv run pytest tests/integration -v`).

## Milestone 3 — delivered

- `integrations/sglang` (`trajectoryos.integrations.sglang`): the single place sampled token
  IDs enter TrajectoryOS. `SGLangClient` calls the engine's native `/generate` with
  `input_ids` (never text) and `return_logprob=True`, reading exact sampled IDs + rollout
  logprobs from `meta_info.output_token_logprobs`. `SGLangToolPolicy` owns the token context:
  the prompt template and tool observations are tokenized once and reported via
  `PolicyTurn.context_token_ids` (recorded `loss_mask=0`); sampled completions are reported
  verbatim (`loss_mask=1`). Text is decoded only to parse `<tool_call>{…}</tool_call>` — never
  re-tokenized into training targets.
- `integrations/slime` (`trajectoryos.integrations.slime`): `trajectory_to_sample_data` splits
  the flattened token stream into prompt/response at the first assistant token so multi-turn
  observations land inside the response with mask 0; every trainable token must carry a real
  rollout logprob (missing → hard `SampleConversionError`, never a silent fill). `to_slime_sample`
  builds a real `slime.utils.types.Sample` in a slime environment. `run_agentic_rollout` composes
  the tested pieces (episode → clean-sandbox verification → composite reward → sample) so scoring
  is atomic with generation. `generate.py` / `reward.py` are the thin slime entry points
  (`--custom-generate-function-path`, `--custom-rm-path`), importing slime/transformers lazily.
- `packages/agents`: `PolicyTurn.context_token_ids` + `run_episode` now record engine-ingested
  context as a `loss_mask=0` environment event, preserving the full token stream.
- `scripts/launch_grpo_qwen3_4b.sh` — runnable slime launch for a real GPU environment
  (Qwen3-4B-Base, GRPO), with the `ALGO` switch already wired for the M5 GSPO experiment.
- Tests (all against mocked inference — no server, no GPU, no slime install):
  `tests/unit/test_sglang_client.py` (token-in/out wire contract via `httpx.MockTransport`),
  `tests/trajectory_correctness/test_sample_conversion.py` (mask/logprob/status invariants),
  `tests/trajectory_correctness/test_agentic_rollout_smoke.py` (end-to-end
  policy → sandbox → clean verification → reward-scored `Sample`, asserting sampled IDs survive
  verbatim and only they are trainable).

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
