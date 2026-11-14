#!/usr/bin/env bash
# Per-agent sweep entry point: `wandb agent` runs this with the sweep's
# --key=value flags; the agent's shell supplies the task workdir and GPU.
#   SWEEP_WORKDIR       task directory for THIS agent (never share one between agents)
#   SWEEP_TURNS_USED    turns_used reported to the environment (default 1)
#   SWEEP_NO_WEAVE=1    skip Weave tracing (wandb metrics only)
#   CUDA_VISIBLE_DEVICES  the GPU this agent owns; inherited by run_task verbatim
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SWEEP_WORKDIR:?set SWEEP_WORKDIR to the task workdir owned by this agent}"
: "${CUDA_VISIBLE_DEVICES:?pin this agent to one GPU with CUDA_VISIBLE_DEVICES}"
extra=()
[ "${SWEEP_NO_WEAVE:-0}" = "1" ] && extra+=(--no-weave)
exec .venv/bin/python -m tracking.wandb_log rollout "$SWEEP_WORKDIR" \
    --turns-used "${SWEEP_TURNS_USED:-1}" "${extra[@]}" "$@"
