#!/usr/bin/env bash
# Parallel W&B sweep agents, one per GPU, for this environment.
# Runs from anywhere; always operates inside AgenticRL-CUDA-kernels/.
#
#   scripts/sweep.sh setup                 # 0. install, load .env, preflight, list GPUs
#   scripts/sweep.sh tasks   [BACKEND] [GPUS...]   # 1. one task workdir per GPU (default: triton, all GPUs)
#   scripts/sweep.sh init                  # 2. create the sweep; saves the ID to .sweep_id
#   scripts/sweep.sh agents  [COUNT] [GPUS...]     # 3. one tmux agent per GPU (default: 16 runs each, all GPUs)
#   scripts/sweep.sh pause | resume | stop # 4. manage the sweep in .sweep_id
#   scripts/sweep.sh status                # tmux sessions + sweep ID
#   scripts/sweep.sh all     [BACKEND] [COUNT]     # 0-3 in one go
#
# Env (from .env via `setup`, or exported): WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT.
# Optional: SWEEP_TURNS_USED (default 1), SWEEP_NO_WEAVE=1, SWEEP_CONFIG (default sweeps/rollout_sweep.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
WANDB=.venv/bin/wandb
SWEEP_CONFIG="${SWEEP_CONFIG:-sweeps/rollout_sweep.yaml}"
SWEEP_ID_FILE=.sweep_id

die() { echo "sweep.sh: $*" >&2; exit 1; }

load_env() {
    [ -f .env ] || die ".env not found (copy .env.example and fill WANDB_API_KEY/WANDB_ENTITY/WANDB_PROJECT)"
    set -a; . ./.env; set +a
    : "${WANDB_API_KEY:?WANDB_API_KEY missing from .env}"
    : "${WANDB_ENTITY:?WANDB_ENTITY missing from .env}"
    : "${WANDB_PROJECT:?WANDB_PROJECT missing from .env}"
}

gpu_indices() {
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true
}

require_sweep_id() {
    # Checked in the parent shell: a die inside $(...) would only exit the subshell.
    [ -s "$SWEEP_ID_FILE" ] || die "no $SWEEP_ID_FILE; run: scripts/sweep.sh init"
}

sweep_path() {
    echo "$WANDB_ENTITY/$WANDB_PROJECT/$(cat "$SWEEP_ID_FILE")"
}

cmd_setup() {
    if [ -x "$PY" ]; then
        "$PY" -m pip install -q -r requirements-gpu.txt -r requirements-tracking.txt
    else
        python3 -m venv .venv && "$PY" -m pip install -q -r requirements-gpu.txt -r requirements-tracking.txt
    fi
    load_env
    "$PY" -m utils.runtime --json --require cuda,triton
    nvidia-smi --query-gpu=index,name,uuid --format=csv
}

cmd_tasks() {
    local backend="${1:-triton}"; shift || true
    local gpus=("$@")
    [ ${#gpus[@]} -gt 0 ] || mapfile -t gpus < <(gpu_indices)
    [ ${#gpus[@]} -gt 0 ] || die "no GPUs found; pass indices explicitly: scripts/sweep.sh tasks $backend 0 1"
    "$PY" -m scripts.make_sweep_tasks --backend "$backend" --gpus "${gpus[@]}" --force
}

cmd_init() {
    load_env
    local out
    out=$("$WANDB" sweep -e "$WANDB_ENTITY" -p "$WANDB_PROJECT" "$SWEEP_CONFIG" 2>&1 | tee /dev/stderr)
    local id
    id=$(printf '%s\n' "$out" | sed -n 's/.*Created sweep with ID: \([A-Za-z0-9]*\).*/\1/p' | tail -1)
    [ -n "$id" ] || die "could not parse the sweep ID from wandb output"
    printf '%s\n' "$id" > "$SWEEP_ID_FILE"
    echo "sweep id $id saved to $SWEEP_ID_FILE ($WANDB_ENTITY/$WANDB_PROJECT/$id)"
}

cmd_agents() {
    load_env
    local count="${1:-16}"; shift || true
    local gpus=("$@")
    [ ${#gpus[@]} -gt 0 ] || mapfile -t gpus < <(gpu_indices)
    [ ${#gpus[@]} -gt 0 ] || die "no GPUs found; pass indices explicitly"
    command -v tmux >/dev/null || die "tmux not installed (apt-get install tmux), or start agents by hand per WANDB_parallel_sweep_agents.md"
    require_sweep_id
    local sweep; sweep=$(sweep_path)
    for g in "${gpus[@]}"; do
        local workdir
        workdir=$(ls -d "$PWD"/sweep_tasks/gpu"$g"/*/ 2>/dev/null | head -1)
        [ -n "$workdir" ] || die "no task workdir for GPU $g; run: scripts/sweep.sh tasks <backend> $g"
        workdir=${workdir%/}
        tmux has-session -t "agent$g" 2>/dev/null && die "tmux session agent$g already exists (tmux kill-session -t agent$g)"
        # One agent == one GPU == one workdir: timing stages never share a device.
        tmux new-session -d -s "agent$g" \
            "export CUDA_VISIBLE_DEVICES=$g SWEEP_WORKDIR='$workdir' SWEEP_TURNS_USED='${SWEEP_TURNS_USED:-1}' SWEEP_NO_WEAVE='${SWEEP_NO_WEAVE:-0}' \
                    WANDB_API_KEY='$WANDB_API_KEY' WANDB_ENTITY='$WANDB_ENTITY' WANDB_PROJECT='$WANDB_PROJECT'; \
             $WANDB agent --count $count $sweep 2>&1 | tee sweep_agent_gpu$g.log"
        echo "started agent$g: GPU $g -> $workdir (count $count); attach: tmux attach -t agent$g"
    done
}

cmd_manage() {
    load_env
    require_sweep_id
    "$WANDB" sweep "--$1" "$(sweep_path)"
}

cmd_status() {
    [ -s "$SWEEP_ID_FILE" ] && echo "sweep: $(cat "$SWEEP_ID_FILE")" || echo "sweep: (none)"
    tmux ls 2>/dev/null | grep '^agent' || echo "no agent sessions"
}

case "${1:-}" in
    setup)   cmd_setup ;;
    tasks)   shift; cmd_tasks "$@" ;;
    init)    cmd_init ;;
    agents)  shift; cmd_agents "$@" ;;
    pause|resume|stop) cmd_manage "$1" ;;
    status)  cmd_status ;;
    all)     cmd_setup; cmd_tasks "${2:-triton}"; cmd_init; cmd_agents "${3:-16}" ;;
    *)       sed -n '2,15p' "$0"; exit 2 ;;
esac
