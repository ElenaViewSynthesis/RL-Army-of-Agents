# TrajectoryOS

> Open-source post-training and agentic-RL platform built around THUDM/slime. SGLang-backed
> rollouts with exact token-ID/loss-mask preservation (no re-tokenization), Docker-sandboxed
> verifiable environments, composable cost-aware rewards, and GRPO/GSPO training — SFT through
> long-context multi-turn agent RL.

An open-source post-training and agentic-RL platform built around
[THUDM/slime](https://github.com/THUDM/slime): SFT, GRPO/GSPO, long-context multi-turn agent
training, synthetic data, verifiable environments, cost-aware routing and deployment.

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Roadmap and status: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## Development

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # install workspace (editable) + dev tools
uv run pytest                  # tests
uv run mypy packages tests     # type check (strict)
uv run ruff format . && uv run ruff check .
```

## Layout

`packages/*` are uv-workspace members under the `trajectoryos.*` namespace
(`trajectoryos.schemas`, `trajectoryos.core`, `trajectoryos.rewards`, ...).
`integrations/slime` is the only place slime's extension interfaces are touched.
See the architecture note for the full tree and the token-ID / loss-mask invariant that
governs all trainable data.
