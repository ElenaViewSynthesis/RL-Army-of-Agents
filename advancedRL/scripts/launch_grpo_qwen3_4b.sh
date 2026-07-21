#!/usr/bin/env bash
# Minimal slime launch for the TrajectoryOS coding-agent GRPO baseline.
#
# Requires a GPU training environment with THUDM/slime installed (e.g. the
# official slime container: zhuzilin/slime). This repo's packages must be
# installed into the same Python environment first:
#
#   pip install -e packages/schemas -e packages/core -e packages/rewards \
#               -e packages/environments -e packages/verifiers -e packages/agents \
#               -e integrations/sglang -e integrations/slime
#
# Cluster-specific values (paths, parallelism) are marked CHANGE-ME.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
ALGO=${ALGO:-grpo}   # grpo | gspo (Milestone 5: only this switch changes)
CKPT_DIR=${CKPT_DIR:-/root/checkpoints/trajectoryos-${ALGO}}   # CHANGE-ME
DATA=${DATA:-/root/data/trajectoryos/repo_bugfix_tasks.jsonl}  # CHANGE-ME

python train.py \
  --advantage-estimator "${ALGO}" \
  --hf-checkpoint "${MODEL}" \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 4096 \
  --rollout-temperature 0.7 \
  --custom-generate-function-path trajectoryos.integrations.slime.generate.generate \
  --custom-rm-path trajectoryos.integrations.slime.reward.reward_func \
  --prompt-data "${DATA}" \
  --input-key prompt \
  --metadata-key metadata \
  --lr 1e-6 \
  --eval-interval 20 \
  --save-interval 50 \
  --save "${CKPT_DIR}" \
  --use-wandb false \
  "$@"
