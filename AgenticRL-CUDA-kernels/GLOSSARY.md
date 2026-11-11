# Glossary

Plain-language definitions of the terms used across this project's README,
validation notes, task briefs and handoffs. Each entry says what the term
means here and, where useful, where it lives in the repository.

**Contents:**
[Rollouts and rewards](#rollouts-and-rewards) ·
[RL algorithms](#rl-algorithms-grpo-dapo-cispo) ·
[GPU programming](#gpu-programming) ·
[Timing, profiling and CUDA graphs](#timing-profiling-and-cuda-graphs) ·
[Verification and tolerances](#verification-and-tolerances) ·
[Reference solutions and testing](#reference-solutions-and-testing) ·
[Task stages and kernels](#task-stages-and-kernels) ·
[Multi-LoRA (Stage 6)](#multi-lora-stage-6) ·
[Figures](#reading-the-figures) ·
[Acronyms](#acronyms)

---

## Rollouts and rewards

**Rollout.** One attempt by the agent: it writes a `model_new.py` (and its
kernels) for a task, and the environment scores it. `run_task()` in
`utils/rollout_env.py` runs one rollout.

**Rollout stages.** A rollout runs as separate child processes, in order,
under one shared deadline (default 180 s):
1. **compile:** build the candidate (nvcc for CUDA, JIT for Triton and the
   DSLs);
2. **verification:** compare the candidate's output with the trusted
   reference on several random inputs;
3. **profiling:** time eager PyTorch, `torch.compile` and the candidate
   with the profiler, and attribute every GPU kernel to its source;
4. **benchmark** (opt-in): repeated timing outside the profiler, with
   optional CUDA-graph replay;
5. **diagnostics** (opt-in): a separate profiler pass that writes a trace,
   a kernel table and a report.

A crash or timeout in one stage ends the rollout with a failure reason; it
cannot take the trainer down.

**Reward rules.** The milestone reward a rollout earns:

| Outcome | Reward | `reward/reason` |
|---|---|---|
| More than 5% faster than both eager and `torch.compile` | 3 | `faster_than_both` |
| More than 5% faster than eager only | 2 | `faster_than_eager` |
| Correct but not faster | 1 | `correct` |
| Correct but provenance below 90% | −1 | `provenance` |
| Wrong output, crash, timeout, turn cap, or no GPU timing | −1 | `verification`, `compile`, `timeout`, `max_turns`, `unmeasured` |

Under the DAPO and CISPO presets, finishing in the last few allowed turns
costs up to 1 point (overlong shaping). Diagnostics never change a reward.

**Provenance.** The share of the candidate's GPU time spent in kernels the
agent actually wrote ("authored"), as opposed to library kernels (cuBLAS,
cuDNN, CUTLASS), PyTorch's own kernels, or memory copies. It is worked out
from kernel names in the profiler. That is a heuristic, not proof of
authorship. A rollout must reach at least 90% to earn a speed reward, which
stops an agent from scoring by just calling `torch.matmul`.

**Authored-only gate (`authored-only-no-library-v1`).** The stricter
task-suite version of provenance: any library time is a violation, and the
authored share must be at least 90% (100% for `gemm_tensor_core`). Approved
2026-11-09; off for the stock rollout.

**Hack fixture / Torch-op hack.** A deliberately dishonest candidate that
computes the right answer with PyTorch operations instead of its own
kernels. It must pass verification and **fail provenance**, which proves
that the provenance gate works.

**Wrong-output fixture.** A candidate that returns a wrong answer on
purpose. It must **fail verification**. Variants: `wrong_ancestor` (Stage 5,
off-by-one tree parent), `wrong_padding` and `wrong_skip` (Stage 6, below).

**GPU UUID.** The globally unique identifier of a physical GPU (for example
`GPU-72d90e64-…`), reported by `nvidia-smi` and recorded in benchmark
records as `bench/device_uuid`. It proves which device a measurement came
from. The read-only results tools drop it from their output.

## RL algorithms: GRPO, DAPO, CISPO

All three are critic-free policy-optimization methods: they score a
**group** of rollouts for the same task and use each rollout's reward
relative to the group (the **advantage**) instead of a learned value
function. Presets live in `utils/config.py`; see the README's
"Configuration" section.

**GRPO (Group Relative Policy Optimization).** Advantage = (reward − group
mean) / group standard deviation. Updates use PPO's clipped probability
ratio. Preset: clip range 0.2/0.2, sequence-level loss averaging, a small KL
penalty (0.001).

**DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization).** GRPO
with four changes:
- a higher upper clip (0.28), so unlikely good tokens can grow faster;
- token-level loss averaging;
- dropping **tied groups**, where every rollout got the same reward and so
  carries no learning signal;
- **overlong shaping**, a soft penalty for using nearly all allowed turns.

No KL penalty.

**CISPO (Clipped IS-weight Policy Optimization, from MiniMax-M1).** Keeps
GRPO's group advantages but clips the **importance-sampling weight** itself
and stops its gradient, instead of clipping the PPO objective. Every token
keeps a gradient. Preset: weight in [0, 5] (`clip_low=1.0`,
`clip_high=4.0`), plus DAPO's tied-group filtering and overlong shaping. No
KL.

**Tied group.** A group in which every rollout got the same reward. Its
advantages are all zero. DAPO and CISPO drop it, and GRPO keeps it with zero
advantage.

## GPU programming

**Kernel.** A function that runs on the GPU across many threads at once.

**CUDA vs Triton.** Two ways to write kernels:
- **CUDA C++** is written per thread. You manage threads, shared memory and
  synchronization yourself, and nvcc compiles it to machine code.
- **Triton** is a Python DSL written per **block** of data (a tile). The
  compiler handles thread mapping, memory coalescing and tensor-core use,
  and kernels are compiled just in time and can be autotuned.

Triton is usually quicker to write and easier to keep correct; CUDA gives
finer control. This project accepts both (plus TileLang, cuTile and CuTe
DSL) behind the same compile → verify → profile pipeline.

**Tensor cores.** Dedicated matrix-multiply units on NVIDIA GPUs. They are
far faster than ordinary arithmetic for fp16/bf16 (and lower precision)
matrix products.

**`tl.dot`.** Triton's block matrix multiply. On supported dtypes it
compiles to tensor-core instructions. A kernel that multiplies element by
element and sums instead skips the tensor cores, which is one reason several
reference solutions are slow.

**Tile and tile reuse.** A tile is a small block of a matrix loaded into
fast on-chip memory (registers or shared memory). **Reuse** means using a
loaded tile for many computations before loading the next, instead of
reading the same data from GPU memory again and again. It is the main
difference between a tuned kernel and a merely correct one.

**GEMM.** General matrix multiply, `C = A·B`.

**GEMM with an epilogue.** A matrix multiply followed, in the same kernel,
by extra elementwise or reduction work on its result: adding a bias,
scaling, an activation, dividing, summing. Fusing the epilogue avoids
writing the product to memory and reading it back.
`gemm_div_sum_scale` (Stage 0, adapted from KernelBench) is a GEMM whose
epilogue divides, sums along rows and scales:
`scale · sum((A · B) / divisor, dim=1)`, an `(M, 1)` fp32 output. Its Triton
solution uses `tl.dot` for the product.

**Reduction.** An operation that combines many values into fewer, such as
a sum, a max or a mean. **Row softmax** (exponentiate and normalize each
row) and **LayerNorm** (normalize each row by its mean and variance) are the
standard reduction tasks, because the whole row must be read before any
output can be written.

**Grid and grid axes.** A kernel launch is a grid of blocks (CUDA) or
programs (Triton) with up to three axes. **Axis 0 allows up to 2³¹ − 1
blocks; axes 1 and 2 allow only 65,535.** Large dimensions, such as a
151,936-entry vocabulary or tens of thousands of tokens, must therefore go
on axis 0. Stage 5's Triton top-k first failed with `invalid argument`
because the vocabulary was on axis 1.

**64-bit offsets.** Memory offsets computed as `int64` rather than
`int32`. An `int32` overflows past 2³¹ (about 2.1 billion elements), which
large tensors reach easily: Stage 5's 32k-context verify attention had an
offset of about 4.3 billion. An overflowed offset points at the wrong
memory and causes an illegal memory access.

**Split-K.** Splitting the summed (K) dimension of a matrix product across
several blocks and adding their partial results afterwards. It keeps a GPU
busy when the output is tiny, for example a LoRA shrink whose output width
is just the rank.

**Compute Sanitizer.** NVIDIA's runtime checker for CUDA programs:
- **memcheck** finds out-of-bounds and misaligned memory accesses;
- **racecheck** finds shared-memory data races;
- **synccheck** finds invalid synchronization;
- **initcheck** finds reads of uninitialized memory.

Its results are diagnostics only and never feed rewards
(`utils/diagnose.py`, `scripts/diagnose_gpu.py`).

**Nsight Systems / Nsight Compute.** NVIDIA's timeline profiler and
per-kernel hardware-counter profiler, used as optional diagnostics.

**CUPTI.** The CUDA Profiling Tools Interface, which PyTorch's profiler
uses to record GPU kernel events.

## Timing, profiling and CUDA graphs

**Eager.** Ordinary PyTorch execution, one operation and one or more
kernel launches at a time. It is the main baseline a candidate must beat.

**`torch.compile`.** PyTorch's compiler. It fuses many small operations
into fewer generated kernels, and is the second, harder baseline.

**Profiler device time.** The reward's timing method: the sum of the
durations of every GPU kernel a call launched, measured by PyTorch's
profiler after two empty "priming" captures. It is per call, in
microseconds.

**Profiler samples sidecar.** Since 2026-11-09 the profiler writes the full
per-kernel event list to a separate JSON file
(`profile/kernel_samples_path`), and keeps only per-kernel totals in its
result. The old single-line record overflowed the 4 MiB output limit for
references that launch thousands of kernels.

**CUDA graph.** A recording of a fixed sequence of GPU kernel launches
that can be replayed with a single launch. It removes most of the CPU-side
launch overhead, which dominates small workloads.

**CUDA-graph capture and replay.** **Capture** runs the model once while
recording its kernels into a graph. **Replay** re-runs exactly those kernels
on exactly the same memory addresses. A graph does not re-read Python code,
so a model with host syncs (`.item()`) or data-dependent shapes cannot be
captured.

**I/O buffers (static input/output buffers).** Because replay reuses the
captured memory addresses, the graph's inputs and outputs live in fixed
("static") tensors. To run on new data you copy it into the static input
buffers, replay, and read the static output buffers.

**`CapturedGraph`.** A small class in `utils/benchmark.py` that holds a
CUDA graph **together with** its static input and output buffers, so they
live exactly as long as the graph.

**The graph-buffer bug.** Before `CapturedGraph`, the benchmark kept only
the graph object. A CUDA graph does not keep its buffers alive, so PyTorch
freed them as soon as the capture function returned, and gave that memory to
other tensors. Every later replay then read and wrote memory it no longer
owned. That produced the Stage 6 smoke's illegal-memory-access errors and
PyTorch "index out of bounds" asserts in 8 of 12 honest benchmark stages
(run `stage6-smoke-20261110T105217Z`). It was fixed on 2026-11-10 with a
regression test; the H200 rerun `stage6-smoke-rerun-20261111T031847Z` checks the
fix on the GPU.

**`--cuda-graphs` forwarding.** The task-suite Modal runner
(`cloud/modal_tasks.py`) passes the `--cuda-graphs` option through to each
case's benchmark stage, so a GPU run records whether each candidate can be
captured and replayed correctly. It needs `--benchmark`.

**Cold start.** The first call after a model is loaded, timed separately
(`--cold-start`) because it includes compilation and autotuning.

**Detached run (`--detach`).** A Modal run that keeps going on Modal's
side even if the local `modal run` client is stopped. That matters here
because this machine's memory reaper sometimes kills idle local processes.

**Frozen source.** An exact copy of the public source taken just before a
GPU run, with a SHA-256 per file (`task-source-manifest.json`). It proves
which code produced a result.

## Verification and tolerances

**Verification.** Running the candidate and the trusted reference on the
same random inputs (5 checks by default) and comparing their outputs.

**Reduced-precision tolerances (upstream `atol`/`rtol` = 1e-2).** The
default comparison is `torch.testing.assert_close(atol=1e-2, rtol=1e-2)`,
inherited from the upstream environment. Each element passes if
`|actual − ref| ≤ atol + rtol · |ref|`, that is, within 0.01 absolute plus
1% relative. That loose bound allows for fp16/bf16 rounding.

**Exact integer outputs.** An approved task-only mode
(`exact_integer_outputs`): integer outputs such as ids, counts or tree
indices must match exactly, with no tolerance.

**The relative bound (Stage 6 gate).** Stage 6's float outputs are judged
against the size of the whole answer instead of element by element:
`max |actual − ref| ≤ c · max |ref|`, with `c = 1e-2` when the case's
activations are bf16 and `c = 5e-3` when they are fp16. The fp32 shrink
buffer uses the same `c` as the case's activations. Any NaN or infinity
fails outright, rows of skipped tokens (`idx = -1`) must be bit-identical
and are left out of `max |ref|`, and an all-zero reference must match
exactly. Approved 2026-11-10 (`relative_error_bounds`,
`utils/lora_verification.py`); Stages 0–5 keep the standard check.

**bf16 / fp16 / fp32.** 16-bit brain float (8 exponent bits: wide range,
low precision), 16-bit IEEE half float (5 exponent bits: narrower range,
more precision), and 32-bit float. Kernels usually store bf16/fp16 and
accumulate in fp32.

## Reference solutions and testing

**Trusted reference.** The PyTorch implementation of a task in its
`model.py` (and `utils/*_reference.py`). It defines the correct output and
is the eager baseline. A candidate cannot change it.

**Reference solution (honest fixture).** A working authored kernel shipped
with a task. It proves the task is solvable and that the gates work. It is
not necessarily fast; see "untuned".

**Untuned.** Correct but not optimized: no tile reuse, no tensor cores, no
pipelining. Several reference solutions are many times slower than eager,
which leaves real headroom for an RL agent.

**Vectorized reference.** A reference written as a few whole-tensor GPU
operations rather than a Python loop over elements. Loops with
`int(tensor)`, `.item()` or `.nonzero()` force a GPU-to-CPU sync on every
step. They make the eager baseline meaninglessly slow, launch thousands of
kernels, and can time out. All Stage 5 and Stage 6 references are
vectorized.

**Small kernel count.** A reference should launch few kernels per call,
because it is the eager baseline: thousands of tiny launches make "faster
than eager" trivially easy and bloat profiler records.

**Loop oracle.** The original, simple, obviously correct loop
implementation, kept only in `tests/fixtures/` (for example
`eagle_reference_loop.py`, `lora_reference_loop.py`). A test checks that
the fast vectorized reference matches it exactly, bit for bit.

**Repeated-forward regression.** A test that calls each model 40 times on
the same input objects and checks that the inputs never change and that
the outputs stay identical. It rules out in-place mutation of inputs.

**CPU interpreter.** `TRITON_INTERPRET=1` runs Triton kernels on the CPU.
It is useful for checking logic without a GPU, but it is slow, cannot
evaluate some features (for example bf16 `tl.dot`), and is never GPU
evidence.

**Smoke / acceptance / stratified.** Case selections for a task:
- **smoke** is the smallest case;
- **acceptance** is the largest;
- **stratified** is a small deterministic set that covers every value of
  every grid axis.

## Task stages and kernels

**Task.** One workload: a trusted `model.py` reference, its input cases,
and the agent's job to write a faster authored `model_new.py`. Stages 0–6
group tasks from easy to hard.

**32k attention.** The two EAGLE-4 attention tasks run over a **32,768
(32k) token** context: 32 sequences, 64 query heads and 8 key/value heads,
head dimension 128, and 64-node draft trees.

**Tuned 32k verify attention (proposed).** A Triton version of tree
verify attention that would use `tl.dot` (tensor cores) for its products
and **reuse key/value tiles** across the 64 tree nodes and the 8 query
heads that share each key/value head. Today each node re-reads the full 32k
context on its own. It would show in the paper that the headroom is
reachable.

**EAGLE-4 / speculative decoding (Stage 5).** Speculative decoding lets a
small draft model propose a **tree** of possible next tokens, which the
large model then verifies in one pass, accepting the longest correct path.
Stage 5 turns EAGLE-4's documented kernels into tasks:
- **candidate top-k:** pick the best tokens from a 151,936-entry vocabulary;
- **tree step:** expand the draft tree;
- **tree attention:** attend over the context plus each node's ancestors;
- **GDN / Mamba2 tree conv and recurrence:** state-space layers along tree
  paths;
- **greedy and sampled acceptance:** decide which draft tokens to keep.

**Chain replay.** An alternate solution for the GDN tree recurrence that
replays each first-child chain of the tree from the request state, instead
of walking nodes in order. It must match the node walk.

## Multi-LoRA (Stage 6)

**LoRA (Low-Rank Adaptation).** Fine-tuning a big weight matrix `W` by
adding a small low-rank update, `W + B·A`. `A` maps the input down to `r`
numbers, and `B` maps them back up. One base model can carry many such
**adapters**.

**Multi-LoRA serving.** Serving many adapters at once on one base model:
each token in a batch may use a different adapter, so the kernels must
apply the right `A` and `B` per token.

**Rank (`r`).** The width of the low-rank bottleneck: how many numbers
`A` produces per token (8 to 256 here, 512 as a stretch). A higher rank
means a more expressive adapter and more work.

**Mixed-rank batch.** One batch whose tokens use adapters of different
ranks, say 8, 64 and 256 together. Kernels must handle each token's real
rank and skip the rest.

**The 32 rank and mixed-rank probes.** The 32 extra CPU-interpreter checks
Codex ran for the Stage 6 gate: high ranks (64 and 256) plus mixed-rank
batches, in fp16 and bf16. All passed; the worst honest case used about 56%
of its allowed error.

**Slot.** A preallocated place in GPU memory that holds one adapter's `A`
and `B`. `max_loras` is the number of slots, and `idx[t]` says which slot
token `t` uses.

**Padding / `max_rank`.** Every slot's matrices are allocated at the
largest rank, rounded up to a multiple of 16, so all slots have one shape.
An adapter of rank 8 in a slot sized for 256 leaves columns 8–255 as
**padding** that must never be read.

**Poisoned padding.** The trusted inputs fill that unused padding with NaN
on purpose. Any kernel that reads padding gets NaN in its output, and since
**any NaN fails verification**, the mistake is caught instead of silently
adding garbage. The `wrong_padding` fixture checks this.

**`idx = -1` (skipped tokens).** A token with no adapter. It must get zero
work and zero writes. Those rows of the output must equal the input exactly,
and the inputs give them large sentinel values so any write is caught. The
`wrong_skip` fixture checks this.

**Shrink.** The first LoRA step, `buf = scale · x · Aᵀ`: it shrinks each
token's hidden vector (thousands of numbers) down to `r` numbers. It is
memory-bound, because the output is tiny.

**Expand.** The second step, `y += buf · Bᵀ`: it expands the `r` numbers
back to the output width and adds them into the base model's output.

**`buf` / fp32 intermediate.** The rank-sized result of shrink, kept in
fp32 between shrink and expand. Keeping it in fp16 loses accuracy, which
is why a `fp16_intermediate` fixture runs as a diagnostic, never as a
gate.

**Slices.** Some layers have several outputs fused in one matrix:
- 1 slice for q, o or down;
- 2 slices for gate_up;
- 3 slices for qkv.

Each slice has its own `A`/`B` and output offset.

**BGMV (Batched Gather Matrix-Vector multiply).** The decode-time kernel
family (one new token per sequence). Each token gathers its adapter's
matrix and does a matrix-vector product. It comes from punica.

**SGMV (Segmented Gather Matrix-Vector multiply).** The prefill/mixed-batch
family. Tokens are grouped into **segments** that share an adapter, so each
segment becomes a small matrix-matrix product that reuses the adapter's
tiles across its tokens.

**Segment metadata.** The per-step bookkeeping SGMV needs: tokens sorted
by adapter, plus each segment's adapter id, start, length, and the count of
active segments. It is built on the GPU with no host sync
(`lora_segment_metadata`).

**Fused SGMV.** Shrink and expand in one kernel, keeping `buf` on chip or
in fp32 scratch so it never round-trips through memory.

**Decode vs prefill.** Decode generates one token per sequence per step,
so batches are small (BGMV sizes). Prefill processes a whole prompt at
once, so there are many tokens per sequence (SGMV sizes).

**punica / S-LoRA.** The two systems whose published designs the Stage 6
kernels follow: punica introduced BGMV/SGMV, and S-LoRA added large-scale
adapter serving and tensor-parallel sharding. Both are Apache-2.0; no code
is copied.

**Tensor parallelism (TP), column- and row-parallel, all-gather /
all-reduce.** TP splits one layer across GPUs. A **column-parallel** layer
splits its outputs, and a **row-parallel** layer splits its inputs and sums
partial results. All-gather and all-reduce are the collective operations
that combine the pieces. TP sharding is in the brief's appendix but **out
of scope** for the Stage 6 tasks.

**AWQ / GPTQ / FP8.** Ways to compress the base model's weights to 4 or 8
bits. LoRA kernels read activations only, so they must work unchanged on a
quantized base.

**TTFT / ITL.** Time to first token and inter-token latency, the serving
metrics the original brief benchmarks (out of scope for the tasks).

## Reading the figures

**Marker key.** What each mark means:
- **squares** are CUDA C++ solutions;
- **filled triangles** are Triton solutions;
- the **hollow triangle (ring)** is the Triton **chain-replay** solution of
  the GDN recurrence;
- in Figure 2, **filled circles** are eager PyTorch and **hollow circles**
  are `torch.compile`.

**Figure 1 (speed vs eager).** Eager time divided by the solution's time,
on a log scale; right of the vertical line is faster. The **winners** are
the tree convs, GDN recurrence and greedy acceptance, about 3× to 230×
faster than eager. Candidate top-k and the 32k attention kernels are
correct but untuned, so slower than eager.

**Figure 2 (time per call).** Every implementation's time per call on a
log scale from microseconds to seconds. The **headroom** is visible where
`torch.compile` beats every reference solution, on 32k attention and
top-k: room an agent could win by tuning.

**Log scale.** An axis where each step is ×10, so values from 7 µs to
1.7 s fit on one chart.

## Acronyms

| Acronym | Meaning |
|---|---|
| ADR | Architecture Decision Record (`Architecture-decision/`) |
| bf16 / fp16 / fp32 | bfloat16 / IEEE half precision / single precision |
| BGMV / SGMV | Batched / Segmented Gather Matrix-Vector multiply |
| CISPO | Clipped IS-weight Policy Optimization |
| CSV | Comma-separated values |
| CUPTI | CUDA Profiling Tools Interface |
| DAPO | Decoupled Clip and Dynamic sAmpling Policy Optimization |
| DSL | Domain-specific language (Triton, TileLang, cuTile, CuTe DSL) |
| GDN | Gated DeltaNet (a linear-attention/state-space layer) |
| GEMM | General matrix multiply |
| GQA | Grouped-query attention (several query heads share a key/value head) |
| GRPO | Group Relative Policy Optimization |
| H200 | NVIDIA Hopper data-center GPU used for all GPU evidence |
| HBM | High-bandwidth memory (the GPU's main memory) |
| IS | Importance sampling |
| ITL / TTFT | Inter-token latency / time to first token |
| KL | Kullback–Leibler divergence penalty |
| LoRA | Low-Rank Adaptation |
| MCP | Model Context Protocol (the planned results connector) |
| MoE | Mixture of experts |
| nvcc | NVIDIA's CUDA C++ compiler |
| OAuth | The sign-in protocol chosen for the connector (via GitHub) |
| PPO | Proximal Policy Optimization (the clipped objective GRPO builds on) |
| PTX / cubin | NVIDIA's intermediate GPU assembly / compiled GPU binary |
| RL | Reinforcement learning |
| SM | Streaming multiprocessor (one GPU compute unit; the H200 has 132) |
| TP | Tensor parallelism |
| UUID | Universally unique identifier (here, the GPU's) |
| W&B | Weights & Biases (experiment tracking) |
