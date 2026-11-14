# Profiling with Nsight Systems: commands for this repo

How to profile a task's kernels with the Nsight Systems CLI (`nsys`), on top of what the harness already
automates (`python -m utils.diagnose`, README section "Nsight profiles"). Everything here is diagnostic:
no profile, report or metric enters a reward. The PyTorch-profiler side, including the reward's own
measurement and a per-kernel script matching `cuda_gpu_kern_sum`, is in
[TORCH_PROFILER_PROFILING.md](TORCH_PROFILER_PROFILING.md).

**What is verified.** Only the harness's default `trace` profile and the `basic` Nsight Compute preset
have H200 evidence (run `20261108T200118Z`). The other commands below come from the Nsight Systems CLI
guide (v2026.6). The Modal image pins **nsys 2026.2.1**, so some commands may be missing or differ there:
`survey`, `query`, `docs` and `finalize` look newer than the pin. Before relying on one, run
`nsys --help` and `nsys <command> --help` on the image. None of these commands has run in this repo yet.

## Per-kernel totals: three tools, one number

On kernel totals the three tools overlap. A script over a `torch.profiler` trace, or
`nsys stats --report cuda_gpu_kern_sum`, gives the same cumulative time per kernel that Graphsignal
reports.

- **`torch.profiler`** is what the reward uses (`utils/profiling.py`). It sums the device time of each
  leaf CUDA kernel over the timed iterations and divides by the iteration count; the per-kernel values
  are the `kernel_us/<name>` fields, and the provenance split groups them by name.
- **`nsys stats --report cuda_gpu_kern_sum`** reads the same kernel records from an Nsight Systems
  report: total time, launch count, and average, median, minimum and maximum duration per kernel name.
- **Graphsignal** is an external profiler this repo does not use; per the statement above, its
  per-kernel view reports the same quantity.

They agree on *what* is summed, not automatically on the *numbers*. Each covers a different window: the
reward profiler times several iterations after warm-up, while the harness's default nsys capture holds
exactly one generated forward. Each tool also adds its own overhead. Compare them on the same capture
window before trusting a difference; this repo has not made that comparison yet.

**CUDA graphs.** By default nsys records a whole graph launch as one item (`--cuda-graph-trace=graph`).
`--cuda-graph-trace=node` times every kernel inside each `cudaGraphLaunch`. The repo's graph-trace run
(`graph-trace-20261112T220601Z`) used `node`, and `utils.diagnose nsys --profile graphs` adds it.

## 1. Collect: `nsys profile`

The harness command builds these for you (README, "Nsight profiles"). Use the raw forms when you want a
switch the harness does not expose. `<cmd>` is the program to profile, for example the diagnostic worker
that `utils.diagnose` launches, or `python -m utils.profiling --json` in a task workdir.

```bash
# Capture only one NVTX range (the harness default). Without registered NVTX strings,
# NSYS_NVTX_PROFILER_REGISTER_ONLY=0 is required for the range name to match.
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=nvtx --nvtx-capture=agenticrl_generated --capture-range-end=stop \
    --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
    --export=sqlite --output <out>/capture --force-overwrite=true <cmd>

# The whole run, with library and OS-runtime tracing, and summary statistics at the end
nsys profile --trace=cuda,nvtx,cublas,cudnn,osrt --sample=none --cpuctxsw=none \
    --stats=true --export=sqlite --output <out>/capture <cmd>

# Kernels inside CUDA graphs, one record per graph node
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --export=sqlite --output <out>/capture <cmd>

# One report per capture range, for the first N ranges with that name
nsys profile --trace=cuda,nvtx --capture-range=nvtx --nvtx-capture=agenticrl_generated \
    --capture-range-end=repeat:5 --output <out>/capture <cmd>

# GPU hardware metrics: list device IDs and metric sets first, then sample one device
nsys profile --gpu-metrics-devices=help
nsys profile --gpu-metrics-set=help
nsys profile --trace=cuda,nvtx --gpu-metrics-devices=<id> --gpu-metrics-frequency=20000 \
    --sample=none --cpuctxsw=none --export=sqlite --output <out>/capture <cmd>

# PyTorch module and autograd ranges, Python call stacks on CUDA calls.
# --cudabacktrace and --python-backtrace need CPU sampling on (--sample=process-tree).
nsys profile --trace=cuda,nvtx,cudnn,cublas,osrt \
    --pytorch=functions-trace-shapes,autograd-nvtx \
    --cudabacktrace=all --python-backtrace=cuda --python-sampling=true --sample=process-tree \
    --output <out>/capture <cmd>

# GPU memory used by CUDA kernels (significant overhead)
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output <out>/capture <cmd>

# Only NVTX events from the default domain (the harness's ranges use it)
nsys profile --trace=cuda,nvtx --nvtx-domain-include=default --output <out>/capture <cmd>

# Is this host able to profile at all?
nsys status --environment
```

## 2. Summarise: `nsys stats`

`nsys stats` takes a `.nsys-rep` or its `.sqlite` export, so it runs directly on the harness's
`<out>/capture.nsys-rep` or `<out>/capture.sqlite`.

```bash
# Cumulative GPU time per kernel name (the number the three tools share)
nsys stats --report cuda_gpu_kern_sum --timeunit usec <out>/capture.nsys-rep

# Several reports at once, written as CSV next to the report
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum \
    --format csv --output . <out>/capture.nsys-rep

# Every kernel launch in time order (start, duration, grid, block, stream, name)
nsys stats --report cuda_gpu_trace --format csv --output . <out>/capture.nsys-rep

# Which reports and formats this nsys version ships
nsys stats --help-reports
nsys stats --help-formats
```

## 3. Look for problems: `nsys analyze`

Expert-system rules over a report, for example idle gaps on the GPU and synchronous CUDA calls:

```bash
nsys analyze --rule gpu_gaps,gpu_time_util,cuda_api_sync,cuda_memcpy_sync <out>/capture.nsys-rep
nsys analyze --help-rules
```

## 4. Export, import, query and survey

```bash
# SQLite export of an existing report, limited to the tables the harness reviewer reads
nsys export --type sqlite --tables 'CUPTI_ACTIVITY_KIND_KERNEL,CUPTI_ACTIVITY_KIND_RUNTIME,NVTX_EVENTS,StringIds' \
    --output <out>/kernels.sqlite <out>/capture.nsys-rep

# Turn a raw .qdstrm into a report when automatic conversion did not run
# (the 2026-11-08 import failure; scripts/recover_nsys.py did this with QdstrmImporter).
# Use the same nsys version that wrote the .qdstrm.
nsys import --input-file <out>/capture.qdstrm --output-file <out>/capture.nsys-rep

# Read-only SQL over a report (needs an aggregate or LIMIT on event tables)
nsys query <out>/capture.nsys-rep --sql \
  "SELECT s.value AS kernel, COUNT(*) AS launches, SUM(k.\"end\" - k.start) AS total_ns
   FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
   GROUP BY kernel ORDER BY total_ns DESC LIMIT 20"

# What a report contains: platform, event counts, diagnostics, applicable recipes
nsys survey <out>/capture.nsys-rep
```

## 5. How this maps onto the harness

| Need | Harness today | Raw CLI above |
|---|---|---|
| One generated forward | `utils.diagnose nsys` (default `trace`) | section 1, first command |
| Eager, compile, warm-up and generated phases | `--profile stages` (NVTX ranges in the worker) | section 1, second command, plus the ranges |
| Kernels inside CUDA graphs | `--profile graphs`; `scripts/graph_trace.py` | `--cuda-graph-trace=node` |
| GPU hardware metrics | `--profile gpu-metrics` (device index from `CUDA_VISIBLE_DEVICES`) | `--gpu-metrics-devices=<id>` |
| Per-kernel totals | reviewer's per-range busy time and provenance split | `nsys stats --report cuda_gpu_kern_sum` |
| Per-kernel counters | `utils.diagnose ncu --preset ...` | (Nsight Compute, not nsys) |

The `stats`, `analyze`, `query` and `survey` commands are not wired into the harness yet; run them by
hand on a capture's `.nsys-rep`. Wiring `nsys stats --report cuda_gpu_kern_sum` into the reviewer is a
natural follow-up.

## 6. Caveats

- **Hopper uses software CUDA trace.** nsys defaults to CUDA hardware trace only from Blackwell on, so
  `--trace=cuda-sw` is not needed on the H200.
- **GPU metric device IDs** are the IDs that `--gpu-metrics-devices=help` reports. The harness passes the
  `nvidia-smi` index of the assigned GPU; whether the two always agree on the image is unverified. The
  alternative is `--gpu-metrics-devices=cuda-visible`, correct when `CUDA_VISIBLE_DEVICES` holds only the
  assigned GPU, as on a one-GPU Modal container.
- **Only the first capture range** starts a collection unless `--capture-range-end=repeat[:N]` is set.
  With `--capture-range-end=stop` (the harness default) the application keeps running after the range;
  the nsys default `stop-shutdown` ends the session.
- **Backtraces need CPU sampling.** `--cudabacktrace` and `--python-backtrace` collect nothing with
  `--sample=none`, so the harness's `backtrace` profile turns sampling on (`--sample=process-tree`).
- **Overhead.** `--cuda-graph-trace=node`, `--cudabacktrace`, `--cuda-memory-usage` and
  `--cuda-trace-all-apis` add significant runtime overhead; never take a timing from such a capture.

## 7. Remote GUI and containerised clusters

Reference material from the Nsight Systems User Guide, for profiling outside this repo's Modal setup.
None of it has been used here. How it relates to this repo:

- **Same version as the image.** The guide's examples use Nsight Systems **2026.2.1**, the version pinned
  in this repo's Modal image (`/usr/local/bin/nsys`).
- **Installed in the image, not bind-mounted.** The Modal image installs `nsys` in the image itself;
  the "inject, don't rebuild" pattern below matters only on a Slurm cluster.
- **Report and GUI versions.** A report from the image must be opened with an Nsight Systems GUI of
  2026.2.1 or newer.
- **CPU sampling.** The harness profiles run with `--sample=none`, except `backtrace`, which needs CPU
  sampling and therefore `perf_event_open`. Whether a Modal container allows that is unverified; check
  `nsys status --environment` inside the container before relying on backtraces.

### Streaming GUI (Nsight Streamer)

A self-hosted Nsight Systems GUI in a Docker container, opened from a web browser. Useful for analysing
reports on a remote server or cluster without copying them to a workstation. Instructions: "Nsight
Streamer for Nsight Systems" on NGC.

### Slurm jobs in containers (Enroot/Pyxis or OCI)

`nsys` must be available **inside** the container that Slurm launches, not only on the host. The
recommended pattern is "inject, don't rebuild": install Nsight Systems once on each compute node,
bind-mount it into the container at launch, and wrap the in-container command with `nsys profile`. The
profiler then upgrades independently of the workload image. Three steps for any runtime:

1. Install the same build at the same path on every compute node, e.g.
   `/opt/nvidia/nsight-systems/2026.2.1` (a shared filesystem works if every node can read it), and
   check it with `/opt/nvidia/nsight-systems/2026.2.1/bin/nsys --version`.
2. Bind-mount that install into the container at the same absolute path.
3. Call `nsys profile` by absolute path, or append its `bin` to `PATH` without replacing the image's
   `PATH`. A wrapper script is needed only for per-rank logic.

**Host kernel requirements.** Profiling from a container still depends on the host:
- `perf_event_paranoid` on the host is permissive enough (typically ≤ 2); check from inside the container
  with `nsys status --environment`;
- the container can call `perf_event_open`: for OCI runtimes `--cap-add=SYS_ADMIN`, `--privileged` or a
  seccomp profile that allows it; Enroot is unprivileged but usually allows it without extra flags
  (verify);
- CUDA driver libraries are provided by the runtime (NVIDIA Container Toolkit, or the Pyxis/Enroot NVIDIA
  hook); the Nsight Systems bind mount does not change that.

**Keep CLI and GUI versions aligned.** Open a report with a GUI of the same or a newer version than the
CLI that recorded it; a newer report in an older GUI is the common failure.

#### Option A: Pyxis/Enroot with `srun --container-mounts`

```bash
# Preflight: the container launch works without profiling
srun --container-image=nvcr.io#nvidia/pytorch:<tag> /bin/true

# Read-only bind mount; nsys called by absolute path, image PATH untouched
srun \
  --container-image=nvcr.io#nvidia/pytorch:<tag> \
  --container-mounts=/opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro \
  /opt/nvidia/nsight-systems/2026.2.1/bin/nsys profile \
    -t cuda,nvtx,mpi \
    -o /reports/run_%q{SLURM_PROCID}_%p \
    python train.py

# nsys on PATH instead, appended inside the container
srun \
  --container-image=nvcr.io#nvidia/pytorch:<tag> \
  --container-mounts=/opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro \
  --export=ALL,NSYS_PATH=/opt/nvidia/nsight-systems/2026.2.1/bin \
  --container-env=NSYS_PATH \
  bash -lc 'PATH="$PATH:$NSYS_PATH"; nsys profile -o /reports/run_%q{SLURM_PROCID}_%p python train.py'
```

For reports on a shared filesystem, also mount the output directory, e.g.
`--container-mounts=...,/scratch/$USER/reports:/reports:rw`, and pass `-o /reports/...`.

#### Option B: Enroot `mounts.d` and `environ.d` (recommended when you do not control `srun`)

Files under `ENROOT_SYSCONF_PATH` (default `/etc/enroot`) apply to every Enroot (and Pyxis) container,
with no image change:

```text
# /etc/enroot/mounts.d/nsys.fstab
/opt/nvidia/nsight-systems/2026.2.1 /opt/nvidia/nsight-systems/2026.2.1 none bind,ro,nosuid,nodev 0 0
```

```text
# /etc/enroot/environ.d/nsys.env
PATH=/opt/nvidia/nsight-systems/2026.2.1/bin:${PATH}
```

#### Option C: Enroot pre-start hook (only when you need logic)

Hooks in `/etc/enroot/hooks.d/*.sh` run with full capabilities before the container switches root and
receive `ENROOT_ROOTFS`, `ENROOT_MOUNTS` and `ENROOT_ENVIRON`:

```bash
#!/usr/bin/env bash
# /etc/enroot/hooks.d/50-nsys.sh: prefer mounts.d / environ.d unless logic is needed.
set -euo pipefail
: "${ENROOT_MOUNTS:?not set}"
: "${ENROOT_ENVIRON:?not set}"
NSYS_HOST_PREFIX="/opt/nvidia/nsight-systems/2026.2.1"
NSYS_IN_CONTAINER="/opt/nvidia/nsight-systems/2026.2.1"
echo "${NSYS_HOST_PREFIX} ${NSYS_IN_CONTAINER} none bind,ro,nosuid,nodev 0 0" >> "${ENROOT_MOUNTS}"
echo "PATH=${NSYS_IN_CONTAINER}/bin:\${PATH}" >> "${ENROOT_ENVIRON}"
```

#### Option D: OCI runtimes (Docker, Podman, containerd, CRI-O)

```bash
docker run --rm --gpus all \
  --cap-add=SYS_ADMIN \
  -v /opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro \
  my-workload:latest \
  /opt/nvidia/nsight-systems/2026.2.1/bin/nsys profile \
    -t cuda,nvtx,mpi \
    -o /reports/run_%p \
    python train.py
```

Or in an OCI `config.json` for runc/crun:

```json
{
  "process": {
    "args": ["/opt/nvidia/nsight-systems/2026.2.1/bin/nsys", "profile",
             "-t", "cuda,nvtx,mpi", "-o", "/reports/run_%p", "python", "train.py"]
  },
  "mounts": [
    {"destination": "/opt/nvidia/nsight-systems/2026.2.1", "type": "bind",
     "source": "/opt/nvidia/nsight-systems/2026.2.1", "options": ["rbind", "ro"]}
  ]
}
```

- `--cap-add=SYS_ADMIN` (or a seccomp profile allowing `perf_event_open`) keeps sampling available.
- Calling `nsys` by absolute path preserves the image's `PATH` and environment; to put it on `PATH`,
  append inside the container rather than replacing `PATH` with `-e` or `process.env`.
- GPU exposure remains the runtime's job (`--gpus all`, the NVIDIA Container Toolkit hook, or the
  `nvidia.com/gpu` device plugin).
- Writing reports to the host needs a second, writable bind mount (`-v /scratch/$USER/reports:/reports:rw`);
  the profiler cannot write through a read-only mount.

#### Wrapping the in-container command

As in a non-container Slurm/MPI job, the launcher runs outside the profiler and `nsys profile` wraps the
per-rank program. Template `-o` with the rank so concurrent ranks do not write the same file:
`%q{SLURM_PROCID}` (Slurm), `%q{OMPI_COMM_WORLD_RANK}` (Open MPI), `%q{PMI_RANK}` (MPICH) or `%p`; `%h`
(hostname) and `%%` (a literal `%`) also work.

```bash
srun [srun args] \
  --container-image=... --container-mounts=... \
  /opt/nvidia/nsight-systems/2026.2.1/bin/nsys profile -t cuda,nvtx,mpi \
    -o /reports/run_%q{SLURM_PROCID}_%p \
    ./myapp [app args]
```

**Profile a subset of ranks** with a wrapper script (`nsys_profile.sh`) inside or mounted into the
container:

```bash
#!/bin/bash
# Profile only the node-local rank 0 (use $SLURM_PROCID for global rank 0).
NSYS_PATH="${NSYS_PATH:-/opt/nvidia/nsight-systems/2026.2.1/bin}"
PATH="${PATH}:${NSYS_PATH}"
if [ "${SLURM_LOCALID}" -eq 0 ]; then
  nsys profile -t cuda,nvtx,mpi \
    -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 \
    -o /reports/run_%q{SLURM_PROCID}_%p \
    "$@"
else
  "$@"
fi
```

```bash
srun --container-image=... \
     --container-mounts=/opt/nvidia/nsight-systems/2026.2.1:/opt/nvidia/nsight-systems/2026.2.1:ro,/scratch/$USER/reports:/reports:rw \
     --export=ALL,NSYS_PATH=/opt/nvidia/nsight-systems/2026.2.1/bin \
     --container-env=NSYS_PATH \
     ./nsys_profile.sh python train.py
```

When only some MPI ranks are profiled, set `NSYS_MPI_STORE_TEAMS_PER_RANK=1`, or the run may hang or fail
with an MPI error; if communicator tracking itself is the problem, `NSYS_MPI_DISABLE_COMMUNICATOR_TRACKING=1`
turns it off.

**GPU and NIC metrics with several ranks per node:** let only one rank collect them, or scope each rank to
its own GPU:

```bash
#!/bin/bash
if [ "${SLURM_LOCALID}" -eq 0 ]; then
  nsys profile --nic-metrics=lf --gpu-metrics-devices=all "$@"
else
  nsys profile "$@"
fi
```

```bash
#!/bin/bash
nsys profile -e CUDA_VISIBLE_DEVICES=${SLURM_LOCALID} \
             --gpu-metrics-devices=${SLURM_LOCALID} "$@"
```

#### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nsys: command not found` inside the container | Bind mount missing, or `nsys` not on `PATH` | Check `ls /opt/nvidia/nsight-systems/<ver>/bin/nsys` inside the container; call `nsys` by absolute path, or append its `bin` to `PATH` inside the container |
| `nsys --version` reports an unexpected release | Wrong host install or a floating symlink | Mount an explicit versioned prefix (`/opt/nvidia/nsight-systems/2026.2.1`), not `.../current` |
| `nsys status --environment` reports sampling disabled, or `Failed to create perf event` | `perf_event_open` blocked by the container or host | OCI: `--cap-add=SYS_ADMIN` or the seccomp profile from "Enable Docker Collection". Enroot: host `perf_event_paranoid` ≤ 2 and no site seccomp shadowing it |
| Reports missing after the job ends | Output path exists only in the container | Point `-o` at a writable bind mount, e.g. `/reports` backed by `/scratch/$USER/reports` |
| Several ranks overwrite one `.nsys-rep` | `-o` not rank-templated | Use `%q{SLURM_PROCID}`, `%q{OMPI_COMM_WORLD_RANK}`, `%q{PMI_RANK}` or `%p` in `-o` |
| Profiler version differs across nodes | Different host installs per node | Pin one version at one path on every node, or use a shared filesystem |
| `pyxis: failed to import docker image with mkdir: cannot create directory '/run/enroot': Permission denied` | Enroot runtime directory missing or not writable for the job user | Make `/run/enroot` exist and be world-traversable with a per-user `/run/enroot/user-<uid>`, or set `ENROOT_RUNTIME_PATH` to a writable path. Quick dev setup: `sudo install -d -m 1777 /run/enroot && sudo install -d -m 700 -o "$USER" -g "$USER" /run/enroot/user-$(id -u)` |
| Enroot/Pyxis import fails with `Could not process JSON input` | Registry endpoint resolved to a web page (docker.io redirects to www.docker.com) | Use an explicit registry: `docker://registry-1.docker.io/library/ubuntu:22.04`, or the Pyxis form `registry-1.docker.io#library/ubuntu:22.04` |
| GUI refuses a report produced in the container | GUI older than the CLI that recorded it | Upgrade the GUI to the CLI's version or newer |
