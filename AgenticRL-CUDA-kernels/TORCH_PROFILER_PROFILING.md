# Profiling with the PyTorch profiler: code for this repo

The companion to [NSIGHT_PROFILING.md](NSIGHT_PROFILING.md). Nsight Systems records a whole process from
outside; the PyTorch profiler (`torch.profiler`) records from inside the Python process, and it is what
the **reward** uses. This file shows how the reward measures, and gives small scripts that reproduce or
extend that measurement. Everything here except `utils/profiling.py` itself is diagnostic: none of it
enters a reward.

**What is verified.** `utils/profiling.py` has H200 evidence across Stages 0–7. The scripts below follow
the same API calls (`torch.profiler.profile`, `ctx.events()`, `event.device_type`, `event.device_time`)
but have not been run as written; they were checked only for Python syntax. The repo's pin is
PyTorch 2.14.1 (`requirements-gpu.txt`).

## 1. How the reward measures (`utils/profiling.py`)

1. **Warm up** the model (`profile_warmup_iters` from `utils/backend.py`: 5 for CUDA C++, 20 for the
   Python DSLs, so JIT compilation finishes first), then `torch.cuda.synchronize()`.
2. **Capture** `iters` forwards inside
   `torch.profiler.profile(activities=[ProfilerActivity.CUDA], record_shapes=False, with_stack=False)`,
   with a final synchronize inside the context.
3. **Keep only device events**: `e.device_type.name == "CUDA"`. CPU operator rows also carry CUDA time,
   and counting them too would count every kernel twice.
4. **Average**: the sum of `event.device_time` (microseconds) divided by `iters` is `t_gen_us`
   (`t_eager_us`, `t_compile_us` for the baselines).
5. **Attribute** each kernel by name (`utils/provenance.py::summarize_events`): authored, library, Torch
   or memory. This gives `provenance/*` and the per-kernel `kernel_us/<name>` values.
6. **Fail closed**: an empty capture gives a null time and reason `empty_profile:<session>`; it is never
   retried and never counted as zero. Two empty "primer" captures run first to warm the profiler itself.

Run it on a task workdir:

```bash
# The reward's measurement, as one JSON line (last line of stdout)
python -m utils.profiling --backend triton --json --iters 10
# The same, plus Chrome traces gen.json, eager.json and compile.json (diagnostic only)
python -m utils.profiling --backend triton --json --iters 10 --trace-dir <dir>
# Baselines first instead of the generated model first (diagnostic ordering check)
python -m utils.profiling --backend triton --json --iters 10 --order eager,compile,gen
```

Open the Chrome traces at `chrome://tracing` or <https://ui.perfetto.dev>.

## 2. Per-kernel totals: the torch side of `nsys stats --report cuda_gpu_kern_sum`

The same rule as the reward (device events only), grouped by kernel name with the columns that nsys's
`cuda_gpu_kern_sum` report prints. Save as `kernel_sum.py`, run it from a task workdir with
`CUDA_AGENT_TASK`, `CUDA_AGENT_TASK_CASE` and `CUDA_AGENT_DEVICE=cuda` set, as the suite does.

```python
"""Per-kernel device-time totals from torch.profiler (diagnostic only)."""
import csv
import statistics
import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from utils.verification import build_inputs, initialize_models


def kernel_sum(model, inputs, warmup=20, iters=10):
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(*inputs)
            torch.cuda.synchronize()
    durations = defaultdict(list)
    for event in prof.events():
        if event.device_type.name == "CUDA":  # leaf device events only, as the reward
            durations[event.name].append(event.device_time)  # microseconds
    total = sum(sum(values) for values in durations.values()) or 1.0
    rows = [{"Time (%)": 100 * sum(values) / total, "Total Time (us)": sum(values),
             "Instances": len(values), "Avg (us)": statistics.fmean(values),
             "Med (us)": statistics.median(values), "Min (us)": min(values),
             "Max (us)": max(values), "Name": name}
            for name, values in durations.items()]
    return sorted(rows, key=lambda row: row["Total Time (us)"], reverse=True)


if __name__ == "__main__":
    reference, generated = initialize_models()
    _, inputs = build_inputs()
    rows = kernel_sum(generated, inputs)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
```

`Total Time (us)` divided by `iters` is the per-forward time that the reward sums; the reward's
`t_gen_us` is the sum of that column divided by `iters`.

## 3. Compare with Nsight Systems on the same kernels

Join the torch totals with `nsys stats --report cuda_gpu_kern_sum --format csv` for the same model.
The nsys column names below are those printed by recent nsys versions; check the header of your CSV
(`nsys stats --help-reports cuda_gpu_kern_sum`) before trusting the join.

```python
"""Side-by-side per-kernel totals: torch.profiler CSV vs nsys cuda_gpu_kern_sum CSV."""
import csv
import sys


def read(path, name_key, total_key, scale):
    with open(path, newline="") as stream:
        return {row[name_key]: (float(row[total_key]) * scale, int(row["Instances"]))
                for row in csv.DictReader(stream)}


torch_rows = read(sys.argv[1], "Name", "Total Time (us)", 1.0)
nsys_rows = read(sys.argv[2], "Name", "Total Time (ns)", 1e-3)  # nsys reports nanoseconds
for name in sorted(set(torch_rows) | set(nsys_rows)):
    t, n = torch_rows.get(name), nsys_rows.get(name)
    print(f"{name[:70]:70}  torch {t[0] if t else float('nan'):10.2f} us x{t[1] if t else 0:<4}"
          f"  nsys {n[0] if n else float('nan'):10.2f} us x{n[1] if n else 0:<4}")
```

They sum the same quantity but rarely the same window: match the iteration count and capture range
before reading a difference as real. Kernel names can differ in formatting between the two tools
(demangled template arguments, spacing); a missing match is not a missing kernel until checked.

## 4. Label phases so both profilers can see them

`torch.profiler.record_function` labels a region in the PyTorch trace; `torch.cuda.nvtx.range` labels it
for Nsight Systems and Nsight Compute. Using both makes one script readable in either tool. The harness's
diagnostic worker uses the NVTX names below (`agenticrl_*`); reusing them keeps traces comparable.

```python
import contextlib

import torch


@contextlib.contextmanager
def phase(name):
    """Label one phase for torch.profiler (record_function) and for Nsight (NVTX)."""
    with torch.profiler.record_function(name), torch.cuda.nvtx.range(name):
        yield


# usage, inside a profile() context or under nsys:
# with phase("agenticrl_eager"):
#     expected = reference(*inputs)
# with phase("agenticrl_generated"):
#     actual = generated(*inputs)
```

Both are host-side markers with no GPU synchronisation. Inside a timed reward path they still cost a
little host time per call; measure that before adding them to a fixture (`NSIGHT_profiling_task.md`).

## 5. A scheduled capture with traces, shapes and memory

For investigation, not timing: a profiler schedule skips the first steps, warms the profiler up, then
records, and writes a Chrome trace for each recorded cycle. Shapes, stacks and memory add overhead.

```python
import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

with torch.no_grad(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=3, active=5, repeat=1),
        on_trace_ready=tensorboard_trace_handler("<trace dir>"),  # one .pt.trace.json per cycle
        record_shapes=True, with_stack=True, profile_memory=True) as prof:
    for _ in range(10):
        generated(*inputs)
        prof.step()  # advances the schedule

# A ranked table. Mixed CPU-op and kernel rows: fine for reading, wrong for summing.
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
# The same grouped by input shapes (needs record_shapes=True)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_device_time_total", row_limit=20))
```

`self_device_time_total` is the column name in recent PyTorch; older versions called it
`self_cuda_time_total`.

## 6. Reproduce the reward's attribution from a capture

The provenance split is a pure function of `(kernel name, device time)` pairs, so any capture can be
classified the same way the reward does:

```python
from utils.backend import agent_kernel_names
from utils.provenance import summarize_events

events = [(e.name, e.device_time) for e in prof.events() if e.device_type.name == "CUDA"]
summary = summarize_events(events, agent_kernel_names(".", "triton"), "triton", iters=10)
print(summary["t_gen_us"], summary["provenance"],
      {k: v for k, v in summary.items() if k.startswith("provenance/")})
```

Name-based attribution is a heuristic, not proof of authorship.

## 7. What each tool is for

| Question | Use |
|---|---|
| The reward's time and provenance | `python -m utils.profiling --json` (sections 1, 6) |
| Per-kernel totals | section 2, or `nsys stats --report cuda_gpu_kern_sum` |
| Which Python line or module launched a kernel | section 5 (`with_stack=True`), or nsys `--profile backtrace` |
| Kernels inside CUDA graph replays | nsys `--cuda-graph-trace=node` (`scripts/graph_trace.py`); not checked with torch.profiler here |
| Phase timeline across eager, compile and generated | nsys `--profile stages`, labelled as in section 4 |
| Per-kernel hardware counters (occupancy, throughput) | Nsight Compute (`utils.diagnose ncu --preset roofline`) |

## 8. Caveats

- **Overhead.** Profiling changes timing; `record_shapes`, `with_stack` and `profile_memory` add more.
  Only `utils/profiling.py`'s fixed configuration feeds a reward.
- **Double counting.** `prof.key_averages()` and `prof.events()` include CPU operator rows that also
  carry device time. Sum only events whose `device_type` is CUDA.
- **Empty captures.** The reward saw one unexplained empty capture of a generated kernel on the H200
  (`stage7-tier1-smoke-20261113T175914Z`, `empty_profile:gen`; the rerun passed). A script that sums an
  empty capture must report "not measured", never zero.
- **CPU runs** record no CUDA events; every device time is unavailable, not zero.
