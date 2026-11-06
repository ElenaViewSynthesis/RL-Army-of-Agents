# AgenticRL CUDA Kernels

**A kernel-generation RL environment that scores what the agent actually wrote: CUDA C++, Triton, TileLang, cuTile or CuTe DSL kernels, gated by where the GPU time went.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)
![PyTorch ≥ 2.3](https://img.shields.io/badge/torch-%E2%89%A52.3-EE4C2C)
![Triton ≥ 3.0](https://img.shields.io/badge/triton-%E2%89%A53.0-6E40C9)
![Tests: 81](https://img.shields.io/badge/tests-81%20%28GPU%20ones%20skip%20without%20CUDA%29-2EA44F)
![GPU results: not measured](https://img.shields.io/badge/GPU%20results-not%20measured-lightgrey)

This project extends the environment released with [CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) ([paper](https://arxiv.org/abs/2602.24286)). It adds three things:

1. **Four kernel backends beyond CUDA C++** (Triton, TileLang, cuTile and CuTe DSL), behind the same compile → verify → profile contract and one backend registry.
2. **A kernel-provenance gate.** The profiler splits device time into *agent-authored*, *library* (cuBLAS / CUTLASS / cuDNN), *Torch* and *memory* kernels. A rollout only earns a speed reward if at least 90% of compute time is in kernels the agent may claim.
3. **GRPO / DAPO reward handling.** Group-relative advantages, tied-group filtering and overlong-turn shaping are exposed as config switches, so a critic-free trainer can consume the discrete milestone reward.

> [!IMPORTANT]
> **No GPU timings have been measured yet.** Development ran on a host without an NVIDIA GPU or `nvcc`, so every eager / torch.compile / CUDA / Triton time, every cuBLAS / CUTLASS / cuDNN share and every provenance percentage below reads *not measured*. CPU runs check correctness only and can never earn a positive reward. The commands to fill these cells on a GPU are in [Reproducing on a GPU](#reproducing-on-a-gpu).

---

## Why provenance matters

Upstream verification blocks `torch.nn.functional.*`, but nothing stops `model_new.py` from calling `torch.matmul`, `torch.sum` or `x @ w.T`. Those calls pass the five correctness checks *and* profile as fast, so the milestone reward cannot tell "the agent wrote a kernel" from "PyTorch ran one".

This environment closes that gap by reading the `torch.profiler` CUDA events it already collects and attributing each kernel to its source. The test suite includes a *Torch-op hack* fixture: it passes correctness and must fail provenance.

## How a rollout is scored

```mermaid
flowchart LR
    A["model_new.py"] --> B{"detect_backend<br/>AST scan, no import"}
    B -->|cuda| C["nvcc + Ninja<br/>cuda_extension.so"]
    B -->|triton| D["Triton JIT<br/>+ autotune"]
    C --> E["verification<br/>5 random inputs, atol/rtol 1e-2"]
    D --> E
    E --> F["profiling<br/>torch.profiler CUDA events"]
    F --> G["provenance<br/>agent / library / torch / memory"]
    G --> H["milestone reward<br/>-1 / 1 / 2 / 3"]
    H --> I["GRPO / DAPO<br/>group advantages"]
```

`run_task(workdir, cfg, turns_used)` runs compile, verification and profiling as **separate child processes** under one wall-clock deadline (default 180 s). A crashing or hanging kernel therefore cannot take the trainer down. The result is a flat, JSON-safe dict ready for `wandb.log`.

| Outcome | Base reward | `reward/reason` |
|---|---|---|
| More than 5% faster than **both** eager and `torch.compile` | **3** | `faster_than_both` |
| More than 5% faster than eager only | **2** | `faster_than_eager` |
| Correct, not faster | **1** | `correct` |
| Provenance below 90% | −1 | `provenance` |
| Wrong output · crash · deadline · turn cap · no GPU timing | −1 | `correctness` · `compile` / `verification` / `profiling` · `timeout` · `max_turns` · `unmeasured` |

Under the DAPO preset, a rollout that finishes inside the last `overlong_buffer_turns` before `max_turns` loses up to 1 point (linear). The base reward never changes.

---

## CUDA vs Triton: how different is it?

Both backends solve the same task (`model.py` → a faster `ModelNew`) and go through the same verification and profiler. What changes is how the kernel is written, built and *credited*.

| | CUDA C++ backend | Triton backend |
|---|---|---|
| **Selected by** | `import cuda_extension` (the default) | `from kernels.triton import …` or `BACKEND = "triton"` |
| **You write** | `__global__` kernels in `kernels/*.cu` and a `*_binding.cpp` registered via `REGISTER_BINDING` | `@triton.jit` functions in `kernels/triton/*.py` and a thin Python wrapper |
| **Programming model** | Per-thread (SIMT): you pick grid and block sizes, indexing, shared memory and syncs | Per-program *blocks*: you index tiles with `tl.arange`, and the compiler handles thread mapping, coalescing and shared memory |
| **Build** | `torch.utils.cpp_extension.load` with nvcc + Ninja → `cuda_extension.so` | JIT on first call, plus `triton.autotune`; no `.so` |
| **Tuning knobs** | Template parameters / launcher `config` switch | `BLOCK_SIZE`, `num_warps`, `num_stages` via `triton.autotune` |
| **Profiler warmup** | 5 iterations | 20 iterations, so autotuning finishes before timing |
| **GEMM** | cuBLAS (permitted for GEMM) or hand-written WMMA/MMA kernels | `tl.dot` only; no `triton.ops`, no third-party kernels |
| **Convolution** | cuDNN (mandatory per upstream skill) | Must be authored |
| **cuBLAS / CUTLASS / cuDNN time** | ✅ Counts as **library**, toward provenance | ❌ Counts as **Torch**, against provenance |
| **Kernel discovery** | Regex over `__global__` declarations (comments stripped) | AST scan for `@triton.jit`, including import aliases |
| **CPU testing** | Not possible (needs nvcc + GPU) | `CUDA_AGENT_DEVICE=cpu TRITON_INTERPRET=1` (NumPy interpreter, one pinned config) |
| **Policy instructions** | [SKILL.md](SKILL.md) (upstream) | [SKILL_TRITON.md](SKILL_TRITON.md) |

The same fused `alpha * a + b` on each backend:

<table>
<tr><th>CUDA (<code>kernels/axpby.cu</code>, upstream)</th><th>Triton (<code>kernels/triton/axpby.py</code>)</th></tr>
<tr><td>

```cuda
template<int THREADS>
__global__ void axpby_kernel(float* out,
    const float* a, const float* b,
    float alpha, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < size; i += stride)
    out[i] = alpha * a[i] + b[i];
}
```

</td><td>

```python
# configs abridged: BLOCK_SIZE 128 / 256 / 512
@triton.autotune(configs=[...], key=["n_elements"])
@triton.jit
def axpby_kernel(a_ptr, b_ptr, out_ptr, alpha,
                 n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, alpha * a + b, mask=mask)
```

</td></tr>
</table>

The CUDA kernel works out one thread's position in a grid-stride loop. The Triton kernel describes a whole block of elements with a mask for the tail, then lets autotuning pick the block size. The CUDA example is float32-only; the Triton one also accepts float16, bfloat16 and float64.

### GEMM: where the backends diverge most

For element-wise kernels like axpby, both backends are judged the same way. For **GEMM-heavy tasks** the difference is large:

- **On CUDA**, the agent may call cuBLAS for GEMM (and cuDNN for convolutions). That time counts as *library* and keeps provenance high. The alternative is a hand-written tensor-core kernel (WMMA/MMA).
- **On Triton**, the GEMM has to be authored with `tl.dot` tiles. Any cuBLAS, CUTLASS or cuDNN kernel in the trace counts *against* the 90% gate, so the agent cannot fall back on a vendor library.
- **The bar is the same on both.** The reward compares against eager PyTorch and `torch.compile` of the reference model, and those normally dispatch matmuls to vendor GEMM libraries. To score 2 or 3, a kernel must beat that by more than 5%. The realistic win is usually **fusion**: computing the GEMM and its epilogue (bias, scaling, reductions) in one kernel instead of round-tripping intermediates through memory.

**Status:** only the axpby example ships today. The optional fused Level-2 GEMM kernel (`gemm_div_sum_scale`, via `tl.dot`) was deferred to prioritise the pipeline and its tests (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)). No GEMM, cuBLAS, CUTLASS or cuDNN times exist yet.

---

## TileLang, cuTile and CuTe DSL backends

Three more Python GPU DSLs plug into the same pipeline through the backend registry in `utils/backend.py`. Each one has an axpby example, a `model_new_<backend>.py` template, a policy doc and an optional requirements file. **None of them has run on a GPU yet.** None ships a CPU interpreter either, so this host only tests their source discovery, backend detection and classification.

| | TileLang | cuTile | CuTe DSL |
|---|---|---|---|
| **Package** | `tilelang` 0.1.15 (tile-ai, third party) | `cuda-tile` 1.8.0 (NVIDIA CUDA Tile) | `nvidia-cutlass-dsl` 4.8.1 (NVIDIA CUTLASS 4.x) |
| **Abstraction** | Tile level with an explicit memory hierarchy: `T.copy`, `T.alloc_shared` / `T.alloc_fragment`, `T.gemm`, `T.Pipelined` | Tile level, Triton-like: `ct.load` / `ct.store` / `ct.gather` / `ct.scatter`, `ct.mma`; the `tileiras` compiler maps tiles to threads | Low level: CuTe layouts, tensors, copy and MMA atoms, explicit thread/value layouts |
| **Authored kernels** | `@T.prim_func` (or `@tilelang.jit`) in `kernels/tilelang/` | `@ct.kernel` in `kernels/cutile/` | `@cute.kernel` in `kernels/cutedsl/`; `@cute.jit` launchers are not credited |
| **Device symbol** | `<prim_func>_kernel` (per TileLang docs) | Undocumented | `kernel_cutlass_<fn>_…_0` by default |
| **GPUs** | SM70–SM120 | Compute capability 8.x–12.x (Hopper needs tileiras 13.3+), driver r580+, Python 3.11+ | Ampere and newer |
| **Example** | `kernels/tilelang/axpby.py`, compiled with `tilelang.compile(target="cuda")` | `kernels/cutile/axpby.py`, launched with `ct.launch` on the current stream | `kernels/cutedsl/axpby.py`, adapted from CUTLASS's elementwise example |
| **Policy · install** | [SKILL_TILELANG.md](SKILL_TILELANG.md) · `requirements-tilelang.txt` | [SKILL_CUTILE.md](SKILL_CUTILE.md) · `requirements-cutile.txt` | [SKILL_CUTEDSL.md](SKILL_CUTEDSL.md) · `requirements-cutedsl.txt` |

**How they relate.** CuTe DSL and cuTile are separate NVIDIA stacks; cuTile is not built on CuTe. TileLang is independent of both, apart from an experimental `cutedsl` compile target. This project doesn't use that target: its TileLang kernels are pinned to `target="cuda"`.

**Provenance rules for all three:**

- **No library credit.** As with Triton, cuBLAS, CUTLASS and cuDNN time counts against the 90% gate.
- **CuTe DSL symbols contain "cutlass".** The classifier would normally treat that as library time, but it checks authored names first, so `kernel_cutlass_axpby_cute_kernel_…` counts as agent work. Tests pin that order.
- **The symbol names above come from documentation, not observed traces.** Record the real profiler names on a GPU before trusting DSL provenance (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)).

Install a backend's extra on top of `requirements.txt`, for example `uv pip install --python .venv/bin/python -r requirements-tilelang.txt`. The extras may conflict with each other or with your Torch build, so use separate environments if needed. `python -m scripts.validate --device cuda` runs each installed DSL's honest and Torch-op hack fixtures, and records any backend it skipped, with the reason, under `unavailable_backends` in `results.json`.

---

## Library time: cuBLAS, CUTLASS and cuDNN

`utils/provenance.py` classifies every CUDA device event by name, in this order:

| Event name contains | CUDA backend | Triton backend |
|---|---|---|
| An authored kernel name (`__global__` or `@triton.jit`) | **agent** ✅ | **agent** ✅ |
| `aten` or `at::native` | torch ❌ | torch ❌ |
| `cublas`, `cutlass` or `cudnn` | **library** ✅ | torch ❌ |
| `memcpy` or `memset` | memory (excluded) | memory (excluded) |
| Anything else | torch ❌ | torch ❌ |

```text
provenance = (agent + library) / (agent + library + torch)      # memory excluded; must be ≥ 0.90
```

**Worked example (made-up durations, not a measurement).** Suppose one iteration spends 80 µs in the agent's kernel, 15 µs in a `cublas` kernel, 5 µs in an `aten` kernel and 10 µs in `memcpy`:

| Backend | Credited | Denominator | Provenance | Gate |
|---|---|---|---|---|
| CUDA | 80 + 15 = 95 µs | 100 µs | 95% | passes |
| Triton | 80 µs | 100 µs | 80% | fails → reward −1, `provenance` |

What `python -m utils.profiling --json` reports:

| Field | Meaning |
|---|---|
| `t_eager_us`, `t_compile_us`, `t_gen_us` | Device time per iteration for eager PyTorch, `torch.compile` and the generated model |
| `provenance` | The gated fraction above |
| `provenance/agent`, `provenance/library`, `provenance/torch`, `provenance/memory` | Each bucket's share of **total** device time, so `provenance/library` is the cuBLAS + CUTLASS + cuDNN share |
| `kernel_us/<event name>` | Per-kernel device time per iteration, which is where individual GEMM, cuBLAS or cuDNN kernels show up |
| `timing_measured` | `false` on CPU, where every timing field is `null` |

The upstream timing line (`Torch Baseline: …us, Torch Compile: …us, CUDA Extension: …us`) is printed byte-for-byte unchanged before the JSON, so existing log parsers keep working.

**Caveats before trusting library numbers** (details in [KNOWN_ISSUES.md](KNOWN_ISSUES.md)):

- **Vendor kernels are matched by name, and that fails closed.** cuBLAS GEMM kernels are often named after the architecture and tile shape rather than the library. Any library kernel whose name lacks `cublas`, `cutlass` or `cudnn` is counted as Torch. Inspect the real `ctx.events()` names on the target GPU before running GEMM tasks.
- **The library bucket can't tell who made the call.** On the CUDA backend, a cuBLAS kernel launched by the agent's extension looks the same as one launched by a stray `torch.matmul` in `model_new.py`.
- **There is no `triton_` prefix rule.** `torch.compile` (Inductor) generates Triton kernels with that prefix too, so authored Triton kernels are credited only through their `@triton.jit` names.
- **Policy text and classifier differ on CUTLASS.** The upstream CUDA skill permits cuBLAS (GEMM only) and cuDNN (convolution only). CUTLASS is not on that list, even though `cutlass`-named kernels are counted as library time.

### CuPy and other CUDA libraries

> **High performance with GPU.** CuPy is an open-source array library for GPU-accelerated computing with Python. CuPy utilizes CUDA Toolkit libraries including cuBLAS, cuRAND, cuSOLVER, cuSPARSE, cuFFT, cuDNN and NCCL to make full use of the GPU architecture.
>
> — [cupy.dev](https://cupy.dev/)

Calling CuPy from a candidate is calling those vendor libraries, so the provenance gate treats it as library or Torch time, never as authored work:

- **Only three library names are recognised.** The classifier's library bucket matches `cublas`, `cutlass` and `cudnn`. Kernels from cuRAND, cuSOLVER, cuSPARSE, cuFFT or NCCL, and CuPy's own generated elementwise and reduction kernels, fall into "anything else" and count as Torch, even on the CUDA backend.
- **On every backend except CUDA, all of it counts against the 90% gate,** including cuBLAS and cuDNN.
- **CuPy is not a dependency of this project.** cuTile's quickstart uses CuPy arrays, but the kernels here take PyTorch tensors.

---

## Triton backend + provenance-gated GRPO/DAPO reward

To use Triton, copy `model_new_triton.py` to a task's `model_new.py` (see [SKILL_TRITON.md](SKILL_TRITON.md)). `--backend auto|cuda|triton` selects compilation. Profiling keeps the upstream timing line and appends per-kernel times and provenance with `--json`; unknown names count as Torch, and memcpy/memset is excluded from the gate denominator.

The case rows come from `python -m scripts.validate --device cpu --timeout-s 600` (raw JSON in [`artifacts/results.json`](artifacts/results.json)). The GPU proof is `python -m unittest -v tests.test_provenance.GPUIntegrationTests`.

| Case | Eager µs | Compile µs | CUDA µs | Triton µs | Provenance % | Outcome |
|---|---|---|---|---|---|---|
| Honest CUDA | not measured | not measured | not measured | — | not measured | GPU test skipped |
| Honest Triton | not measured | not measured | — | not measured | not measured | CPU correctness; unmeasured reward |
| Torch hack | not measured | not measured | — | not measured | not measured | CPU correctness; GPU gate test skipped |
| Timeout | not measured | not measured | — | not measured | not measured | timeout |
| Max turns | not measured | not measured | — | not measured | not measured | max_turns |
| Overlong (DAPO) | not measured | not measured | — | not measured | not measured | shaping unit-tested; GPU check skipped |

Device-time breakdown by bucket, from `provenance/*` in `profiling --json`:

| Kernel | Agent share | cuBLAS / CUTLASS / cuDNN share | Torch share | Memory share |
|---|---|---|---|---|
| axpby, CUDA backend | not measured | not measured | not measured | not measured |
| axpby, Triton backend | not measured | not measured | not measured | not measured |
| Level-2 GEMM | not implemented (deferred) | — | — | — |

The synthetic honest / honest / hack group below has rewards 3, 3 and −1, and also comes from `scripts.validate`. These are arithmetic fixtures, not GPU results.

| Preset | Triton advantage | CUDA advantage | Hack advantage | Tied group |
|---|---|---|---|---|
| GRPO (synthetic) | 0.707106 | 0.707106 | −1.414213 | kept, zero advantage |
| DAPO (synthetic) | 0.707106 | 0.707106 | −1.414213 | dropped |

---

## Benchmark mode (opt-in)

`python -m utils.benchmark --json` samples steady-state latency outside the profiler, for `eager`, `compile` and `generated`. Flags: `--warmup-iters` (10), `--rounds` (6, must be a multiple of 3), `--iters-per-round` (10), `--backend`, `--seed`, `--cuda-graphs`. Each impl runs **prepare** (one timed cold call covering compile/JIT/autotune, kept as `bench/<impl>/prepare_s`), then `warmup_iters` untimed calls, then the rotated rounds — never inside the statistics.

Each round reports two per-iteration timings: `event_us_*` (CUDA events on the current stream) and `wall_us_*` (a `torch.cuda.synchronize()`-bracketed host clock, which also includes CPU dispatch). Statistics over rounds are `n`, `median`, `min`, `max`, `p25`, `p75` and `cv` (`bench/<impl>/{event,wall}_us_*` plus `_cv`). `bench/order` records the rotation and `bench/seed` the seed; `bench/device_*` keys identify the GPU. `bench/generated/event_wall_ratio` well below 1 points to un-joined side-stream work (see the stream contract below). With `--cuda-graphs`, each impl is also captured into a `torch.cuda.CUDAGraph` after warmup and replayed in the same rotation (`graph_capturable`, `graph_error`, `graph_replay_ok`, `graph_us_*`, `graph_all_capturable`). Capture failure is reported, never a crash, and graph timings are diagnostics only — the reward still uses `t_*_us` from `utils.profiling`.

**Stream/device contract.** `ModelNew.forward` must enqueue all device work on `torch.cuda.current_stream()` of its inputs' device, or make that stream wait on any side streams before returning. Event timings only cover work ordered on that stream, while the wall timing waits for every stream, so un-joined side-stream work shows up as an event time far below the wall time. The same rule is given to the policy in [SKILL_TRITON.md](SKILL_TRITON.md) and documented in `utils/benchmark.py`.

`RolloutConfig.benchmark` adds a fourth `run_task` stage (after profiling, same deadline) that merges the flat `bench/*` keys into the result; `bench_warmup_iters`/`bench_rounds`/`bench_iters_per_round`/`cuda_graphs` tune it. With the flag off, the stage argv, result keys and reward are byte-for-byte unchanged. On CPU or a missing GPU the record is just `{"bench/available": false, "bench/unavailable_reason": ...}` — no timings are ever fabricated. A benchmark stage that crashes, times out or returns malformed output degrades the same way (`bench/available: false` with the reason); it can never change `env/*` or the reward.

---

## GPU runtime (cloud)

**No GPU run has happened yet.** The pinned cloud stack is [requirements-gpu.txt](requirements-gpu.txt): the `nvidia/cuda:13.3.1-devel-ubuntu24.04` image (system nvcc + headers at `/usr/local/cuda`), Python 3.12, `torch==2.14.1+cu130`, `triton`, `ninja`, `cuda-tile==1.8.0` with `cuda-toolkit[tileiras]>=13.3,<13.4`, `tilelang==0.1.15` and `nvidia-cutlass-dsl[cu13]==4.8.1`. Install it with **pip** — uv's first-index strategy cannot resolve the `+cu130` wheel (`UV_INDEX_STRATEGY=unsafe-best-match` also works). Hopper (sm_90: H100/H200) needs tileiras **13.3**, which also matches Modal's CUDA 13.3 driver API.

Preflight on any GPU host: `python -m utils.runtime --json --require cuda,triton,tilelang,cutile,cutedsl` — it prints a flat `runtime/*` record (driver, nvcc, ninja, tileiras, torch/triton/cuda-tile/tilelang/cutlass-dsl versions, GPU UUID) and exits non-zero when a profile is missing; it never imports torch.

On a bare GPU host:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-gpu.txt
python -m utils.runtime --json --require cuda,triton,tilelang,cutile,cutedsl
python -m scripts.record_kernel_names --device cuda
python -m unittest -v tests.test_provenance tests.test_dsl_kernels tests.test_benchmark.GPUBenchmarkTests
python -m scripts.validate --device cuda
```

On Modal: `modal run cloud/modal_runtime.py` (preflight) or `--tests` (require gate, stock pipeline, `record_kernel_names`, GPU suites, validate); `MODAL_GPU` selects the GPU (default H200). Logs and `results.json`/`environment.json`/`kernel_names.json` land in the `agenticrl-cuda-artifacts` Volume under `/artifacts/<run_id>/`.

---

## Quick start

Python 3.11+ on **Linux or WSL2** is required. The floor is 3.11 rather than 3.10 so a future cuTile Python backend fits without a second runtime: cuTile supports Python 3.11–3.15, compute capability 8.x–12.x (Hopper H100/H200 is 9.0), NVIDIA driver r580+ and CUDA Toolkit 13.1+, and takes PyTorch tensors as kernel arguments ([prerequisites](https://docs.nvidia.com/cuda/cutile-python/quickstart.html#prerequisites)). Process isolation uses POSIX process groups, so the process tests fail on native Windows. The CUDA path also needs `nvcc` and Ninja on `PATH` (the cloud image provides both).

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"

# Upstream CUDA pipeline (stock axpby)
bash utils/compile.sh && python3 -m utils.verification && python3 -m utils.profiling

# Full scored rollout of this folder as a task
python -m utils.rollout_env . --algo grpo --turns-used 1
```

Keep the repository root's `model_new.py` as shipped: tests assert the upstream files are byte-identical. To try Triton, copy the folder to a task workdir and use `model_new_triton.py` as its `model_new.py`:

```bash
python -m utils.compile --backend triton
python -m utils.verification
python -m utils.profiling --backend triton --json --iters 10
```

### CPU-only correctness

```bash
uv pip install --python .venv/bin/python 'torch>=2.3' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python 'triton>=3.0' numpy   # NumPy powers the Triton interpreter

python -m unittest discover -v                        # GPU-only tests skip explicitly
python -m scripts.validate --device cpu --timeout-s 600
```

On a cold host the first Triton interpreter compile can take minutes, so local CPU validation uses `--timeout-s 600`. The production default stays 180 s.

### Reproducing on a GPU

These are the only runs that can replace the *not measured* cells above:

```bash
python -m unittest -v tests.test_provenance.GPUIntegrationTests
python -m scripts.validate --device cuda
```

The GPU tests require honest CUDA and Triton to pass correctness *and* provenance, and the Torch-op hack to pass correctness but fail provenance and rank last under both presets.

---

## Configuration: GRPO vs DAPO

`RolloutConfig` (`utils/config.py`) is pure Python and never imports Torch. Every field is a `--kebab-case` flag, and presets are applied *before* overrides, so `--algo dapo --clip-high 0.3` changes one knob only.

| Toggle | GRPO preset | DAPO preset | Owned by |
|---|---|---|---|
| Clip range (`clip_low` / `clip_high`) | 0.2 / 0.2 | 0.2 / 0.28 | trainer |
| Loss aggregation (`loss_agg`) | `seq` | `token` | trainer |
| Tied-group filtering (`filter_tied_groups`) | keep (zero advantage) | drop | **environment** |
| Overlong shaping (`overlong_buffer_turns`) | 0 (hard cutoff) | 4 (linear penalty) | **environment** |
| `kl_coef` | 0.001 | 0.0 | trainer |

Other defaults: `provenance_threshold=0.90`, `speedup_margin=0.05`, `max_turns=20`, `max_context_tokens=32768`, `timeout_s=180`, `group_size=8`.

> For a controlled comparison that differs only in the four toggles, hold `--kl-coef`, seeds and budgets equal across both runs.

**Phase 2: Modal rollouts.** Call `utils.rollout_env.run_task(workdir, cfg, turns_used)` with `RolloutConfig.from_args(...)`. Recommended first-run flags: `--max-turns 20 --max-context-tokens 32768 --timeout-s 180 --group-size 8`. The agent loop enforces turns and context. The environment also checks the turn count and enforces the shared wall-clock deadline. CPU runs, or runs without GPU measurements, return null fields and reason `unmeasured`, never a positive reward.

**Phase 3: tracking.** The W&B namespaces are `env/`, `reward/`, `adv/`, `turns/` and `meta/`; drop `log/*` before logging. `RolloutConfig.to_dict()` is the MLflow parameter set. No tracking library is imported here.

---

## Repository layout

```text
AgenticRL-CUDA-kernels/
├── model.py · model_new.py            # upstream reference model + stock CUDA ModelNew (unchanged)
├── binding.cpp · binding_registry.h   # upstream fixed bindings (unchanged)
├── model_new_triton.py                # Triton ModelNew template
├── model_new_<dsl>.py                 # TileLang / cuTile / CuTe DSL templates
├── kernels/
│   ├── axpby.cu · axpby_binding.cpp   # upstream CUDA example (unchanged)
│   ├── triton/axpby.py                # authored, autotuned Triton kernel
│   └── {tilelang,cutile,cutedsl}/     # DSL axpby examples (GPU only, not yet run)
├── utils/
│   ├── compile.py · compile.sh        # --backend auto|cuda|triton|tilelang|cutile|cutedsl
│   ├── verification.py                # 5 random-input checks, atol/rtol 1e-2
│   ├── profiling.py                   # upstream timing line + --json provenance
│   ├── benchmark.py · stats.py        # opt-in rotated benchmark; flat bench/* stats
│   ├── backend.py                     # backend registry + authored-kernel discovery (AST/regex, no imports)
│   ├── provenance.py                  # agent / library / torch / memory classifier
│   ├── reward.py                      # milestone reward, provenance gate, overlong shaping
│   ├── grpo.py                        # group advantages, tied-group filter, batch stats
│   ├── config.py                      # RolloutConfig, GRPO/DAPO presets, CLI
│   └── rollout_env.py                 # run_task(): isolated, deadline-bounded evaluation
├── scripts/validate.py                # reproducible fixtures → artifacts/results.json
├── tests/                             # 81 tests; GPU-only ones skip without CUDA or the DSL package
├── artifacts/                         # raw logs, environment record, upstream checksums
├── SKILL*.md                          # policy instructions, one per backend
├── requirements*.txt                  # torch + triton; optional per-DSL extras
└── VALIDATION.md · KNOWN_ISSUES.md    # evidence and open gaps
```

## Scope and limitations

- **Environment only.** There is no trainer, policy weights, agent loop, Modal orchestration, MLflow or W&B code in this phase.
- **Not a hostile-code sandbox.** Process groups isolate crashes and kill descendants on timeout. A production Modal sandbox still needs filesystem, network and resource limits and a container-level deadline.
- **Shared-GPU timing skews rewards.** Avoid concurrent profiling on one sandbox GPU, because contention changes both milestone rewards and group-relative advantages.
- **Cold starts count against the deadline.** nvcc / JIT / autotune and `torch.compile` warmup all fall inside the 180 s deadline; Level-3 tasks may need more.

Full evidence, raw command output and what each test establishes are in [VALIDATION.md](VALIDATION.md). Open issues are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Background reading

- [Triton_provenance_GRPO_task.md](Triton_provenance_GRPO_task.md): the task brief this implementation follows
- [CUDA_on_Modal.md](CUDA_on_Modal.md) · [GPU-acceler.md](GPU-acceler.md): the CUDA stack on Modal and GPU-accelerated libraries
- Upstream: [BytedTsinghua-SIA/CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) at commit `473025c` (checksums in [artifacts/upstream.json](artifacts/upstream.json))
