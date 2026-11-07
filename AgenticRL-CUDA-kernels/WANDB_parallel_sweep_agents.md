# Parallel sweep agents (W&B) — multi-GPU focus

Reference copy of the W&B "Parallelize agents" page
(source: <https://docs.coreweave.com/products/wandb/sweeps/parallelize-agents>,
fetched 2026-11-06), kept next to the other
runtime guides (`CUDA_on_Modal.md`, `GPU-acceler.md`). The multi-GPU
procedure is the one that matters for this project; the multi-CPU tabs are
kept for completeness.

## How this maps to AgenticRL-CUDA-kernels

- One `wandb agent` per GPU, each pinned with `CUDA_VISIBLE_DEVICES`, is the
  contention-safe layout for this environment: `run_task` inherits
  `CUDA_VISIBLE_DEVICES` verbatim, and its timing stages (profiling, the
  opt-in benchmark) must never share a GPU with another measuring rollout —
  contention changes both the milestone reward and the group-relative signal
  (see `KNOWN_ISSUES.md`, "Modal cold starts and contention"). Per-GPU
  serialisation of measurement stages is still future work (spec Step 6), so
  until then **never start two agents on the same device index**.
- The sweep's `program` is the tracked rollout entry point,
  `python -m tracking.wandb_log rollout <workdir> [RolloutConfig flags]`; every
  `RolloutConfig` field already has a `--kebab-case` flag, so sweep parameters
  map 1:1 onto `cfg/*` (e.g. `--algo`, `--group-size`, `--provenance-threshold`,
  `--bench-rounds`). Each agent run logs the flat `env/*`, `reward/*`, `meta/*`,
  `bench/*` keys to the W&B project named by `WANDB_ENTITY`/`WANDB_PROJECT` in `.env`.
- On a Modal host, `MODAL_GPU="H200:2"` (or more) gives one container with
  several GPUs; launch one agent per index inside it exactly as below. The
  sweep must be initialised first (`wandb sweep <config.yaml>` → `[SWEEP-ID]`).
- Use the project venv (`.venv/bin/wandb`, `.venv/bin/python`); the system
  interpreter is PEP 668 "externally managed" and refuses `pip install`.

## Runbook: parallel agents on a multi-GPU host

Scripted version — `scripts/sweep.sh` wraps steps 0–4 below, reads
`WANDB_ENTITY`/`WANDB_PROJECT`/`WANDB_API_KEY` from `.env`, saves the sweep ID
to `.sweep_id` (gitignored) and starts one tmux agent per GPU:

```bash
scripts/sweep.sh setup                  # 0. install, load .env, preflight, list GPUs
scripts/sweep.sh tasks triton 0 1       # 1. one workdir per GPU (omit indices = all GPUs)
scripts/sweep.sh init                   # 2. wandb sweep → .sweep_id
scripts/sweep.sh agents 16              # 3. tmux session agent<N> per GPU, 16 runs each
scripts/sweep.sh status | pause | resume | stop
scripts/sweep.sh all triton 16          # 0-3 in one go
```

Files involved: `sweeps/rollout_sweep.yaml` (sweep definition — parameters
are `RolloutConfig` fields), `scripts/sweep_rollout.sh` (what each agent
executes; reads `SWEEP_WORKDIR` and `CUDA_VISIBLE_DEVICES` from the agent's
shell), `scripts/make_sweep_tasks.py` (one isolated task workdir per GPU),
`scripts/sweep.sh` (the orchestration above). The manual steps follow.
Verified locally end to end with `WANDB_MODE=offline` on the CPU interpreter;
the GPU-host run has not happened yet.

```bash
# 0. One-time setup on the GPU host (same for Modal "H200:2" containers)
cd AgenticRL-CUDA-kernels
pip install -r requirements-gpu.txt -r requirements-tracking.txt   # bare host; in the repo venv use .venv/bin/pip
set -a && . ./.env && set +a   # WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT (see .env.example; never commit .env)
.venv/bin/python -m utils.runtime --json --require cuda,triton       # preflight: driver, nvcc, GPU UUIDs
nvidia-smi --query-gpu=index,name,uuid --format=csv                  # the indices you will pin below

# 1. One task workdir per GPU (never share a workdir between agents)
.venv/bin/python -m scripts.make_sweep_tasks --backend triton --gpus 0 1
#   prints: export CUDA_VISIBLE_DEVICES=0 SWEEP_WORKDIR=.../sweep_tasks/gpu0/triton
#           export CUDA_VISIBLE_DEVICES=1 SWEEP_WORKDIR=.../sweep_tasks/gpu1/triton

# 2. Initialise the sweep once; copy the SWEEP-ID it prints
.venv/bin/wandb sweep -e "$WANDB_ENTITY" -p "$WANDB_PROJECT" sweeps/rollout_sweep.yaml

# 3. One agent per GPU, each in its own terminal (or tmux pane), pinned with
#    CUDA_VISIBLE_DEVICES and pointed at its own workdir. --count bounds the
#    runs per agent; SWEEP_TURNS_USED feeds the turn-based shaping.
# terminal A
export CUDA_VISIBLE_DEVICES=0 SWEEP_WORKDIR="$PWD/sweep_tasks/gpu0/triton" SWEEP_TURNS_USED=1
.venv/bin/wandb agent --count 16 "$WANDB_ENTITY/$WANDB_PROJECT/[SWEEP-ID]"
# terminal B
export CUDA_VISIBLE_DEVICES=1 SWEEP_WORKDIR="$PWD/sweep_tasks/gpu1/triton" SWEEP_TURNS_USED=1
.venv/bin/wandb agent --count 16 "$WANDB_ENTITY/$WANDB_PROJECT/[SWEEP-ID]"

# 3b. Same thing from one shell with tmux (N = number of GPUs)
SWEEP="$WANDB_ENTITY/$WANDB_PROJECT/[SWEEP-ID]"
for g in 0 1; do
  tmux new-session -d -s "agent$g" \
    "export CUDA_VISIBLE_DEVICES=$g SWEEP_WORKDIR=$PWD/sweep_tasks/gpu$g/triton SWEEP_TURNS_USED=1; \
     .venv/bin/wandb agent --count 16 $SWEEP 2>&1 | tee sweep_agent_gpu$g.log"
done
tmux ls                      # attach with: tmux attach -t agent0

# 4. Manage the sweep
.venv/bin/wandb sweep --pause  "$WANDB_ENTITY/$WANDB_PROJECT/[SWEEP-ID]"
.venv/bin/wandb sweep --resume "$WANDB_ENTITY/$WANDB_PROJECT/[SWEEP-ID]"
.venv/bin/wandb sweep --stop   "$WANDB_ENTITY/$WANDB_PROJECT/[SWEEP-ID]"   # agents exit after their current run

# Dry run of exactly what an agent executes (no W&B account needed):
CUDA_VISIBLE_DEVICES=0 SWEEP_WORKDIR="$PWD/sweep_tasks/gpu0/triton" SWEEP_NO_WEAVE=1 WANDB_MODE=offline \
  bash scripts/sweep_rollout.sh --algo=dapo --seed=1 --iters=10 --benchmark --bench-rounds=6 --timeout-s=180
```

Notes: `wandb agent` passes sweep parameters as `--key=value`; booleans arrive
as a bare `--flag` because the command uses `${args_no_boolean_flags}`. Inside
each rollout, `run_task` sees only the pinned GPU, and `bench/device_uuid` /
`runtime/gpu0_uuid` record which device actually measured. Other backends:
`--backend tilelang|cutile|cutedsl|cuda` in `make_sweep_tasks`; the sweep
yaml does not set `backend`, so it is detected from the workdir.

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.coreweave.com/llms.txt
> Use this file to discover all available pages before exploring further.

> Parallelize sweep agents on multi-core or multi-GPU machines.

# Parallelize agents

Parallelize your sweep agents on a multi-core or multi-GPU machine to run multiple sweep runs at the same time. Parallelization reduces the time needed to explore your hyperparameter space. This page shows you how to launch parallel agents on multi-CPU and multi-GPU machines so you can use all available compute on a single host.

Before you start, you must initialize your sweep. For more information, see [Initialize sweeps](https://docs.coreweave.com/products/wandb/sweeps/initialize-sweeps.md).

## Parallelize on a multi-CPU machine

The following tabs describe how to run multiple sweep agents in parallel on the same machine. Parallel agents are most useful on a machine with multiple CPU cores, where each agent can run on its own core. Choose the tab for the CLI or a Jupyter Notebook based on your workflow.

> [tab] CLI
>
> Use the [`wandb agent`](https://docs.coreweave.com/products/wandb/ref/cli/wandb-agent) command to parallelize your sweep agent across multiple CPUs with the terminal. Provide the sweep ID that Weights & Biases returned when you [initialized the sweep](https://docs.coreweave.com/products/wandb/sweeps/initialize-sweeps.md).
>
> 1. Open more than one terminal window on your local machine.
> 2. Copy and paste the following code snippet and replace `[SWEEP-ID]` with your sweep ID:
>
>    ```bash
>    wandb agent [SWEEP-ID]
>    ```

> [tab] Notebook
>
> Use the W\&B Python SDK library to parallelize your sweep agent across multiple CPUs within Jupyter Notebooks. Provide the sweep ID that Weights & Biases returned when you [initialized the sweep](https://docs.coreweave.com/products/wandb/sweeps/initialize-sweeps.md) and set the `function` parameter to the name of the function the sweep executes:
>
> 1. Open more than one notebook.
> 2. Copy and paste the sweep ID on multiple notebooks to parallelize a sweep. For example, if you have the sweep ID stored in a variable called `sweep_id` and the name of the function is `function_name`, paste the following code snippet on multiple notebooks to parallelize your sweep:
>
>    ```python
>    wandb.agent(sweep_id=sweep_id, function=function_name)
>    ```

## Parallelize on a multi-GPU machine

Run sweep agents in parallel across multiple GPUs on the same machine by using `CUDA_VISIBLE_DEVICES` to assign each [`wandb agent`](https://docs.coreweave.com/products/wandb/ref/cli/wandb-agent) to a different GPU, so the agents run in parallel without competing for the same device.

To follow this procedure, you need:

* A machine with more than one GPU.
* The NVIDIA CUDA Toolkit installed.
* A terminal.

For example, suppose you have two NVIDIA GPUs on your local machine. Replace `[SWEEP-ID]` in each command with the sweep ID that Weights & Biases returns when you initialize a sweep:

1. In one terminal, set `CUDA_VISIBLE_DEVICES` to `0` and start an agent:

   ```bash
   CUDA_VISIBLE_DEVICES=0 wandb agent [SWEEP-ID]
   ```

2. In a separate terminal, set `CUDA_VISIBLE_DEVICES` to `1` and start another agent with the same sweep ID:

   ```bash
   CUDA_VISIBLE_DEVICES=1 wandb agent [SWEEP-ID]
   ```
