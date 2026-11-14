# Nsight Perf SDK for AI: commands for this repo

[NVIDIA Nsight Perf SDK for AI](https://github.com/NVIDIA/nsight-perf-sdk-for-ai) ("Beta Preview", no
tagged release; this repo pins commit `bd5d5901c6c9b72704245dd0f6e8e1a822bda527`) is an always-on
tracer. A long-running service (`nvidia-nsight-perf`) records the CUDA API, GPU work, PM counters,
power, context switches and NVTX ranges from **any** CUDA process on the node. Consumers then pull a
window of that data into a Trace Stream Database, which exports to Perfetto. Unlike `nsys profile`,
it does not launch the workload: the service runs first and the workload runs unchanged. Like
everything in [NSIGHT_PROFILING.md](NSIGHT_PROFILING.md), it is diagnostic only and never feeds a
reward.

**Requirements (from the SDK README), and what they mean here**

| Requirement | This repo |
|---|---|
| Blackwell or Rubin GPU | **Not the H200** (Hopper). On Modal: `B200` or `B300` |
| Ubuntu 24.04, amd64 or aarch64 | the task images already use Ubuntu 24.04 |
| CUDA Toolkit 13.5+ | `nvidia/cuda:13.5.0-devel-ubuntu24.04` (the task image is 13.3.1) |
| Driver 620+, or kernel driver 580.95.05+ with the CUDA 13.5 compatibility package | the image ships `/usr/local/cuda/compat`; GR-context tracing needs the service's `libcuda.so` to match the kernel driver |
| CMake 3.24+, C++17 compiler | Ubuntu 24.04 `cmake` and `build-essential` |
| Counter permission for all metrics | otherwise disable the PM / HW-trace providers (below) |

The service must share a PID namespace with the workload or be an ancestor of it (the README says to
use the host PID namespace for containers). Only one service may own a GPU, and GPU PM counters are
exclusive. So do not run it while `ncu`, or a `nsys` run with GPU metrics, uses the same GPU.

## 1. Build

```bash
git clone https://github.com/NVIDIA/nsight-perf-sdk-for-ai.git
cd nsight-perf-sdk-for-ai
git checkout bd5d5901c6c9b72704245dd0f6e8e1a822bda527
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
export NSIGHT_PERF_BIN="$PWD/build/bin"   # nvidia-nsight-perf and libnvperf_grfx_host.so are downloaded, not built
```

## 2. Start the service

```bash
# Root always runs in global scope (/run/nvidia/nsight-perf/); a normal user gets a
# user-local service under $XDG_RUNTIME_DIR/nvidia/nsight-perf/.
"$NSIGHT_PERF_BIN/nvidia-nsight-perf" &
"$NSIGHT_PERF_BIN/nvtrace_poll_until_service_ready"

# Without GPU PM counters and GPU hardware trace (no counter permission, or another
# profiler owns the counters): CUDA API and NVTX tracing only.
NSIGHT_PERF_GPU_PM_ENABLE=0 NSIGHT_PERF_GPU_HW_TRACE_ENABLE=0 "$NSIGHT_PERF_BIN/nvidia-nsight-perf" &
"$NSIGHT_PERF_BIN/nvtrace_poll_until_service_ready" --ignore gpu_pm,gpu_hwt

# Configuration: export the built-in YAML, edit it, validate it, then select it
"$NSIGHT_PERF_BIN/nvidia-nsight-perf" --export-default-config-only ./nvidia-nsight-perf.yaml
"$NSIGHT_PERF_BIN/nvidia-nsight-perf" --config ./nvidia-nsight-perf.yaml --validate-config
NVIDIA_NSIGHT_PERF_CONFIG_PATH=./nvidia-nsight-perf.yaml "$NSIGHT_PERF_BIN/nvidia-nsight-perf" &
```

The YAML sets the PM sampling rate (`GpuPmCounters.pmSampleRateHz`, at most 20 kHz recommended), the
history length (`maxHistoryDurationMsec`; the built-in history is 402 ms) and the metric configs.

## 3. Run a task's workload while a consumer records

The workload is any CUDA process. `scripts/nsight_perf_workload.py` runs one task's generated model:
it checks the output once, then runs timed forwards, each inside an NVTX push/pop range named
`agenticrl_generated`. It uses no `torch.profiler`, so it does not compete with the tracer.

```bash
# A task workdir (here relu_squared, honest Triton fixture) and its case
python -c "from utils.task_suite import scaffold; scaffold('relu_squared', 'triton', 'honest', '/tmp/relu-triton')"
export CUDA_AGENT_TASK=relu_squared CUDA_AGENT_TASK_CASE=b4096_h14336_bfloat16 CUDA_AGENT_DEVICE=cuda TRITON_INTERPRET=0

# Fixed window: start the consumer first, then the workload inside the window
"$NSIGHT_PERF_BIN/nvtrace_streaming_consumer" --output-dir ./tsf --duration-ms 30000 &
sleep 3
(cd /tmp/relu-triton && PYTHONPATH="$REPO" python -m scripts.nsight_perf_workload --iters 20000)
wait

# Lookback: the last N ms the service still holds (bounded by maxHistoryDurationMsec)
"$NSIGHT_PERF_BIN/nvtrace_snapshot_consumer" --output-dir ./snapshot --duration-ms 1000
```

On a 580-series kernel driver, the workload can load the 13.5 user-mode driver from the compatibility
package. The service keeps the host-matched `libcuda.so`, because GR-context tracing needs it:

```bash
(cd /tmp/relu-triton && LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH \
    PYTHONPATH="$REPO" python -m scripts.nsight_perf_workload --iters 20000)
```

## 4. Export and open

```bash
# Each consumer writes <hostname>.<timestamp>/ under its --output-dir
for db in ./tsf/*/ ./snapshot/*/; do
    "$NSIGHT_PERF_BIN/nvtrace_perfetto_trace_exporter" --tsf-database-dir "$db"
done
# Open the .perfetto-trace.zip at https://ui.perfetto.dev ; gzip -9 very large uncompressed traces
```

## 5. On Modal

`cloud/modal_nsight_perf.py` runs sections 1–4 plus a `utils.diagnose nsys` capture of the same case
on a Blackwell GPU, in its own image (CUDA 13.5 devel with nsys, plus the SDK built at the pinned
commit). The H200 task image is unchanged. It is a paid GPU run:

```bash
MODAL_GPU=B200 modal run --detach cloud/modal_nsight_perf.py --run-id <id> \
    [--task relu_squared --case b4096_h14336_bfloat16 --backend triton --iters 20000 --duration-ms 30000]
modal volume get agenticrl-cuda-artifacts /<id>/ <local dir>
```

Each step's stdout and stderr land in `/<id>/<step>.std{out,err}.log`, and `summary.json` records
them. The app records the driver, then tries the full service. If the service doesn't come up, it
falls back to a reduced service without GPU PM counters and hardware trace. With a pre-620 driver, it
adds a second pass that runs the workload on the compatibility driver. Use the Triton backend there:
the image builds CUDA C++ extensions for `sm_100` (`TORCH_CUDA_ARCH_LIST=10.0`), but this repo's CUDA
fixtures are only validated on `sm_90`.

## 6. First run on Modal: the service did not start

Run `nsight-perf-20261114T095539Z` (B200, `relu_squared`, Triton, `b4096_h14336_bfloat16`). Evidence is in
`artifacts/runs/nsight-perf-20261114T095539Z/` (logs and JSON; the binary nsys capture was left out).

| Step | Outcome |
|---|---|
| Image build | SDK built cleanly at the pinned commit on CUDA 13.5.0; the image's nsys is **2026.4.0** (not the task image's 2026.2.1); `/usr/local/cuda/compat` holds `libcuda.so.620.70.06` |
| Host | NVIDIA B200, compute capability 10.0, driver **610.57.04**; the container kernel reports `4.19.0-gvisor` (Modal runs containers under gVisor) |
| Workload alone | verified; 20,000 forwards in 766.6 ms of CUDA-event time, 38.3 µs per forward (diagnostic only) |
| Service, full | `nvtrace_poll_until_service_ready` exited 1 after 60 s; the service printed nothing to stdout or stderr |
| Service, no GPU PM / HW trace | the same: exit 1 after 60 s, no output |
| Consumers, export, compat pass | not run (no ready service) |
| `utils.diagnose nsys` on the same case | **passed**: one authored kernel `relu_squared_triton_chunks`, 37,920 ns; all four `nsys stats` reports and four `nsys analyze` rules exited 0; `cuda_gpu_kern_sum` total = SQLite total (37,920 ns) |

The `nsys analyze` stages exited 0 with empty stdout, so whether that means "no findings" or that output
went elsewhere is not established. For scale, 37,920 ns for one forward that reads and writes
2 × 4096 × 14336 bf16 values (234,881,024 bytes) is about 6.19 TB/s of logical traffic (derived, not
measured by a counter).

**Why the service did not start is not established.** It printed nothing, so the run cannot
distinguish between the candidate causes:
- the driver: 610.57.04 is below 620, and the service loads the host-matched `libcuda.so`, so the
  compatibility package (which only helps the workload) does not help it;
- gVisor: the service needs the host PID namespace and driver event interfaces, which a gVisor
  sandbox may not expose even with GPU passthrough;
- something unlogged in the service itself.

A next attempt would capture `nvidia-nsight-perf --help`, `--validate-config`, and the service under
`strace`. It is also a paid GPU run.

## 7. Known limitations (SDK README)

- NVTX start/end ranges are not supported; push/pop ranges are (`torch.cuda.nvtx.range_push/pop`).
- On the 620 driver, NVTX used in a parent before `fork()` and again in the child can trace wrongly,
  hang or crash: use `spawn` or `forkserver` for PyTorch `DataLoader` workers.
- Kernels that start before the observation window may be missing.
- A user-mode driver that does not match the kernel driver can break GR-context tracing (missing
  kernel timings or correlation).
- Batched async memcpy (`cuMemcpyBatchAsync_v2`) can appear as one event.
- Traces contain host PIDs; check before sharing them.
