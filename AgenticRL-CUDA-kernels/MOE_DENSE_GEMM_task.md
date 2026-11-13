# MASTER RESEARCH ENGINEERING PROMPT
## High-Performance MoE-as-Dense-GEMM Kernels for NVIDIA Blackwell — CUDA, Triton, CuTe DSL, and TensorRT-LLM

### ROLE

Act as a Principal NVIDIA GPU Kernel Engineer, CUDA/Tensor Core Microarchitecture Specialist, Triton Compiler Engineer, and ML Systems Research Scientist with expertise in:

- NVIDIA Blackwell SM100/SM103 microarchitecture
- CuTe DSL, CUTLASS, CUDA C++, PTX, and Triton
- TMA, `tcgen05.mma`, persistent CTAs, warp specialisation, and asynchronous pipelines
- NVFP4/NVFP8 quantisation and block-scaled matrix multiplication
- Tensor-parallel and expert-parallel Mixture-of-Experts inference
- Roofline modelling, occupancy analysis, and GPU performance profiling
- TensorRT-LLM, SGLang, PyTorch, and distributed LLM inference

Your task is to **research, implement, optimise, test, and benchmark novel high-performance MoE-as-Dense-GEMM GPU kernels**, starting from NVIDIA's published DENSEGEMM design.

The objective is not to reproduce an existing kernel verbatim. It is to develop an independently validated implementation and investigate performance improvements suitable for an ML systems research publication.

Every proposed optimisation must have an explicit performance hypothesis, correctness specification, benchmark, and reproducible experimental result.

---

# 1. PRIMARY TECHNICAL REFERENCE

Read and analyse this article completely:

https://nvidia.github.io/TensorRT-LLM/1.3.0rc15/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html

Also inspect the TensorRT-LLM repository:

https://github.com/NVIDIA/TensorRT-LLM

Relevant upstream components include:

- `fused_moe_densegemm.py` / `cutedsl/dense_gemm.py`
- `create_moe.py`
- `moe_as_dense_gemm/fc1.py`
- `moe_as_dense_gemm/fc2.py`
- Custom operation registration
- NVFP4 weight packing and scale-factor transformations
- MoE backend selection
- Existing grouped-GEMM baselines

Do not assume these files are in their historical directories. Resolve their paths against the actual inspected TensorRT-LLM tag/commit.

Identify the exact source version used for the reference implementation and distinguish it from newer upstream revisions.

Additional references:

- https://github.com/NVIDIA/cutlass
- https://github.com/triton-lang/triton
- https://github.com/meta-pytorch/tritonbench
- https://github.com/gpu-mode/reference-kernels
- https://github.com/flashinfer-ai/flashinfer

## Core research question

**Under which combinations of token count, expert count, Top-K routing, precision, GPU architecture, and weight-layout strategy does dense MoE execution outperform grouped GEMM despite performing redundant arithmetic?**

Secondary questions:

1. Can dense MoE FC1/FC2 be further fused without sacrificing numerical correctness?
2. Can inactive experts be skipped without reintroducing expensive dispatch overhead?
3. Can router computation overlap with expert computation?
4. Can a hardware-aware hybrid dispatcher outperform both pure dense and pure grouped approaches?
5. Can numerical discrepancies introduced by fused alpha scaling be eliminated or controlled?
6. How does the optimal algorithm change between A100, H100, and B200?
7. Which optimisations remain beneficial in realistic LLM serving and RL rollout inference workloads?

---

# 2. MATHEMATICAL SPECIFICATION

Consider a routed MoE layer with:

- `M`: number of input tokens
- `E`: number of routed experts
- `k`: number of selected experts per token
- `H`: hidden dimension
- `I`: expert intermediate dimension
- `X`: input tensor, shape `[M, H]`

Each expert contains:

- `W1[e]`: gate projection `[I, H]`
- `W3[e]`: up projection `[I, H]`
- `W2[e]`: down projection `[H, I]`

Define the expert computation:

\[
f_e(x)=W_{2,e}\left(\operatorname{SiLU}(W_{1,e}x)\odot W_{3,e}x\right)
\]

The routed MoE output is:

\[
y_t=\sum_{e\in S_t}p_{t,e}f_e(x_t)
\]

Here, \(S_t\) contains the Top-K selected experts and \(p_{t,e}\) is the selected expert's routing weight.

Define a dense routing mask:

\[
\alpha_{t,e}=
\begin{cases}
p_{t,e}, & e\in S_t\\
0, & \text{otherwise}
\end{cases}
\]

The dense formulation becomes:

\[
Y=\sum_{e=1}^{E}\alpha_{:,e}\odot F_e(X)
\]

Investigate representing FC1 and FC2 as large matrix multiplications.

## Dense FC1

Pack the gate/up weights into a dense matrix:

\[
W_{13}\in\mathbb{R}^{(2EI)\times H}
\]

Compute:

\[
G=XW_{13}^{T}
\]

Apply SwiGLU with correct gate/up indexing:

\[
Z_{t,e,i}=\operatorname{SiLU}(G^{gate}_{t,e,i})G^{up}_{t,e,i}
\]

The implementation must support a hardware-optimised interleaved weight layout, not just a naive concatenation.

## Dense FC2

Flatten the expert activations and concatenate the down-projection weights:

\[
Z_{\mathrm{flat}}\in\mathbb{R}^{M\times(EI)}
\]

\[
W_{2,\mathrm{dense}}\in\mathbb{R}^{H\times(EI)}
\]

Then compute:

\[
Y=(\alpha\odot Z)_{\mathrm{flat}}W_{2,\mathrm{dense}}^{T}
\]

The scaling must be applied at the correct precision and position.

Investigate moving alpha scaling from a separate tensor operation into the FC2 GEMM mainloop.

**Important:** Moving alpha scaling before or after quantisation can change numerical results. Mathematical equivalence in real arithmetic does not establish equivalence under NVFP4 quantisation.

---

# 3. FIRST IMPLEMENT A TRUSTED PYTORCH REFERENCE

Before building CUDA kernels, implement a small, transparent PyTorch reference that reproduces the mathematical MoE output.

Start from this conceptual baseline:

```python
import torch
import torch.nn.functional as F

def moe_dense_reference(
    x,
    w1,
    w3,
    w2,
    topk_ids,
    topk_weights,
):
    # x:  [M, H]
    # w1: [E, I, H]
    # w3: [E, I, H]
    # w2: [E, H, I]
    # topk_ids, topk_weights: [M, k]

    M, H = x.shape
    E, I, _ = w1.shape

    assert w3.shape == (E, I, H)
    assert w2.shape == (E, H, I)

    alpha = x.new_zeros((M, E))
    alpha.scatter_(1, topk_ids, topk_weights)

    # Simple correctness-first layout.
    # Optimised kernels may pack weights differently.
    w13 = torch.stack(
        (w1, w3), dim=2
    ).reshape(E * 2 * I, H)

    fc1 = (x @ w13.T).reshape(M, E, 2, I)

    gate = fc1[:, :, 0, :]
    up = fc1[:, :, 1, :]

    hidden = F.silu(gate) * up
    hidden = hidden * alpha.unsqueeze(-1)

    w2_dense = (
        w2.permute(1, 0, 2)
          .contiguous()
          .reshape(H, E * I)
    )

    return hidden.reshape(M, E * I) @ w2_dense.T
```

This is a **correctness baseline, not a performance implementation**.

Require unique, valid Top-K expert indices unless duplicate-ID aggregation is explicitly supported.

Build a separate sparse grouped-expert PyTorch reference that computes only selected experts. Verify that both mathematical formulations agree within dtype-appropriate tolerances.

Test biases, routing-score normalisation, shared-expert behaviour, and weight-sharding conventions only when they are enabled in the explicit workload specification.

Do not silently mix different MoE architectures.

---

# 4. IMPLEMENTATION CURRICULUM

Create a progressive sequence of GPU kernel challenges.

Each level must compile, execute, pass correctness tests, and produce real device-side performance measurements before advancing.

## LEVEL 1 — Fundamental CUDA and Triton Kernels

Implement:

### Task 1.1 — Alpha Mask Generation

Write CUDA and Triton kernels to transform Top-K indices and routing scores into dense alpha masks.

Input:

- `topk_ids[M, k]`
- `topk_scores[M, k]`

Output:

- `alpha[M, E]`

Optimise:

- GPU-resident execution
- Coalesced memory accesses
- Avoiding unnecessary intermediate allocations
- Correct handling of masked and unselected experts
- Fusion with preceding routing operations where possible

Benchmark against PyTorch scatter.

### Task 1.2 — Expert-Aware SwiGLU

Implement:

\[
\operatorname{SwiGLU}(a,b)=\operatorname{SiLU}(a)\odot b
\]

Support an input of shape `[M, E, 2, I]`.

Optimise indexing for both simple packed and hardware-interleaved FC1 layouts.

### Task 1.3 — Alpha Scaling Fusion

Implement a kernel that performs:

\[
Z_{t,e,i}\leftarrow Z_{t,e,i}\alpha_{t,e}
\]

Compare separate scaling against fusion into the SwiGLU epilogue or FC2 load path.

## LEVEL 2 — Dense GEMM Implementation

### Task 2.1 — Correctness-First Dense FC1

Implement:

\[
[M,H]\times[H,2EI]\rightarrow[M,2EI]
\]

Begin with BF16 or FP16.

Provide:

- Tiled Triton GEMM
- CUDA C++ GEMM
- Correct boundary masking
- Accumulation precision controls
- Explicit stride handling
- Reproducible layout transformations

### Task 2.2 — Fused Dense FC1 + SwiGLU

Fuse:

1. GEMM accumulation
2. Gate/up extraction
3. SiLU activation
4. Element-wise multiplication

Minimise intermediate global-memory traffic.

### Task 2.3 — Dense FC2 with Expert Masking

Implement:

\[
[M,EI]\times[EI,H]\rightarrow[M,H]
\]

Apply the correct per-token-per-expert alpha during the GEMM mainloop or at an independently validated equivalent point.

Do not require materialisation of a full scaled activation tensor unless justified by performance measurements.

### Task 2.4 — Fused FC1/FC2 Evaluation

Measure individual kernel latency and end-to-end layer latency.

Separate prepacking costs from steady-state execution, but report both where relevant.

## LEVEL 3 — Blackwell Hardware-Aware Kernels

Implement native NVFP4 support on compatible Blackwell GPUs.

Investigate:

- Block-scaled FP4 Tensor Core operations
- `tcgen05.mma`
- TMA global-to-shared transfers
- Persistent CTA scheduling
- Warp-group or warp-specialised pipelines as supported
- Producer/consumer synchronisation
- Shared-memory swizzles
- Register pressure reduction
- CTA cluster configurations
- MMA tile selection
- Quantisation scale loading and transformations
- Persistent tile scheduling
- Fused FC1 activation epilogues

Use CuTe DSL/CUTLASS to expose Blackwell-specific features where Triton does not provide an equivalent or sufficiently performant implementation.

Initial FC1 candidate MMA tiles:

- `(128,128)`
- `(128,256)`
- `(256,256)`

Initial FC2 candidate MMA tiles:

- `(128,64)`
- `(128,128)`
- `(128,256)`

Investigate CTA cluster configurations:

- `(1,1)`
- `(1,2)`
- `(1,4)`
- `(2,1)`, where supported

These are candidate configurations, not universal optima.

Match the kernel's quantisation semantics, scale-factor representation, memory layout, and Tensor Core instruction requirements to the target Blackwell architecture.

Validate NVFP4 pack/unpack behaviour against an independent reference.

Do not claim hardware-native NVFP4 acceleration on A100 or H100.

---

# 5. ADVANCED RESEARCH TASKS

Develop the following independent optimisation experiments.

## Experiment A — Hybrid Dense / Grouped GEMM

Hypothesis:

Dense GEMM is preferable in an intermediate memory-bound token range, while grouped GEMM is better for tiny or large batches.

Implement:

- Dense execution
- Grouped execution
- Runtime selection between them

Explore a dispatcher based on:

\[
d^*=\arg\min_{d\in\{\mathrm{dense},\mathrm{grouped}\}}
\widehat{T}_d(M,E,k,H,I,\mathrm{dtype},\mathrm{GPU})
\]

Build latency models from measured profiling data.

Avoid choosing the backend using expensive online profiling on the critical inference path.

Include hysteresis or stable decision boundaries if noisy measurements cause frequent backend switching.

Investigate whether expert-load statistics improve decisions beyond token count alone.

## Experiment B — Skipping Zero-Load Experts

Identify experts that receive no tokens in the current batch.

Develop a GPU-side compacted active-expert representation.

Investigate whether active-expert compaction and selective weight loading can improve latency for very small token batches.

Compare:

- Full dense execution
- Grouped GEMM
- Active-expert-only execution
- Hybrid dense execution over compacted experts

Measure the compaction overhead separately.

## Experiment C — Router/TopK Overlap

Investigate scheduling router computations concurrently with the independent parts of dense FC1.

Prototype:

- Stream A: FC1 GEMM
- Stream B: Router GEMM, TopK, alpha generation
- Synchronisation before the first operation that consumes alpha

Prove correctness using CUDA events, dependency tracing, and race checks.

Measure exposed versus overlapped critical-path latency.

Do not assume two kernels execute concurrently merely because they use different streams.

## Experiment D — Fused Shared Expert

Extend the dense representation from `E` routed experts to `E+1` experts.

Assign the shared expert an alpha weight of 1.

Investigate combining routed and shared FC1/FC2 into the same persistent GEMM mainloop.

Measure whether removing the standalone shared-expert execution offsets the additional arithmetic and memory traffic.

Retain an unfused shared-expert baseline.

## Experiment E — Numerical Closure of Fused FC2 Alpha

Compare two approaches:

1. Applying alpha inside FC2.
2. Folding alpha into FC1's post-activation output before the relevant quantisation boundary.

Test:

- FP32 reference
- BF16/FP16
- NVFP4
- Different routing-score magnitudes
- Zero scores
- Large activation magnitudes
- Quantisation saturation
- Subnormal values
- Different scale-factor granularities

Measure absolute error, relative error, and output distribution differences.

Identify when the two formulations diverge and whether additional scaling or higher-precision intermediate operations can close the gap.

Never promote a faster fused implementation that fails the defined module-level numerical contract.

## Experiment F — Workload-Specialised Expert Routing

Investigate routing distributions:

- Uniform random
- Highly skewed
- Hot-expert concentration
- Near-uniform expert utilisation
- Adversarially unbalanced Top-K assignment

Determine whether routing entropy, selected-expert count, and assignment concentration predict the optimal GEMM strategy.

Use these measurements to inform the hybrid dispatch policy.

---

# 6. GPU BENCHMARK SPECIFICATION

Construct a controlled benchmark matrix.

## Token counts

Test at minimum:

`M = [1, 8, 16, 32, 48, 64, 96, 128, 192, 208, 224, 256, 272, 320, 384, 512, 1024]`

## Expert counts

`E = [8, 16, 32, 64, 128, 256]`

## Top-K

`k = [1, 2, 4, 8]`

Require `k <= E`.

## Precision

- FP32 reference
- BF16
- FP16
- FP8 where hardware and baseline support exist
- NVFP4 on compatible Blackwell hardware

## Model dimensions

Cover representative per-rank MoE shapes, including the NVIDIA DeepSeek-V3-style reference:

- `H = 7168`
- `E = 256`
- `k = 8`
- `I` derived from the selected tensor-parallel configuration

Also include smaller configurations that fit A100/H100 memory budgets.

Do not exhaustively execute the Cartesian product when combinations are infeasible. Define a justified representative matrix.

## Benchmark implementations

Compare:

1. PyTorch sparse MoE reference
2. PyTorch dense MoE reference
3. CUDA dense GEMM implementation
4. Triton dense GEMM implementation
5. CuTe DSL Blackwell implementation
6. CUTLASS grouped-GEMM baseline
7. TensorRT-LLM grouped MoE where available
8. TensorRT-LLM DENSEGEMM where available
9. Hybrid runtime-dispatched implementation

Ensure the baselines use the same mathematical operation, precision contract, routing distribution, and workload shape.

Record when a backend cannot support a configuration.

---

# 7. ROOFLINE AND PERFORMANCE MODEL

Develop a theoretical performance model before optimisation.

For dense FC1, approximately:

\[
F_{\mathrm{FC1}}=4MEIH
\]

For dense FC2:

\[
F_{\mathrm{FC2}}=2MEIH
\]

Therefore:

\[
F_{\mathrm{dense,total}}\approx6MEIH
\]

The selected-expert arithmetic for the corresponding sparse calculation is approximately:

\[
F_{\mathrm{sparse,total}}\approx6MkIH
\]

Thus, ignoring router, activation, packing, and other overheads:

\[
\frac{F_{\mathrm{dense}}}{F_{\mathrm{sparse}}}\approx\frac{E}{k}
\]

For 256 experts and Top-8 routing, this represents approximately 32 times as much expert GEMM arithmetic.

This is an arithmetic-count comparison, not a predicted latency ratio.

For the simplified memory-bound FC1 model:

\[
AI\approx\frac{2M}{b}
\]

Where `b` is the weight payload size in bytes per element.

For packed NVFP4, the payload term gives `b = 0.5`, excluding scale-factor overhead.

Derive the architecture-specific arithmetic-intensity crossover:

\[
AI_{\mathrm{crit}}=\frac{P_{\mathrm{peak}}}{BW_{\mathrm{HBM}}}
\]

Use sustained or measured hardware limits where available, and separately report theoretical peak values.

Predict the approximate crossover between memory-bound and compute-bound dense execution.

Compare this estimate against measured Nsight Compute results.

Quantify modelling error from:

- Quantisation scales
- Activation memory traffic
- Weight-cache effects
- GEMM tile inefficiency
- Launch overhead
- Register pressure
- Tensor Core utilisation
- Tactic changes

---

# 8. PROFILING AND OPTIMISATION

Use NVIDIA Nsight Compute and Nsight Systems.

Record:

- Kernel execution latency
- Full MoE module latency
- SM active cycles
- Tensor Core activity
- DRAM throughput
- L2 throughput
- Achieved occupancy
- Shared-memory throughput
- Register count
- Shared-memory footprint
- TMA activity
- Warp stalls
- Launch overhead
- Quantisation cost
- Routing and Top-K cost
- Synchronisation overhead

Generate a profile-based explanation for every material performance gain or regression.

The analysis must distinguish:

**Kernel improvements:** faster FC1 or FC2 device execution.

**Fusion improvements:** elimination of intermediate memory traffic or launches.

**Scheduling improvements:** reduced exposed critical-path latency.

**Algorithmic improvements:** choosing a better dense, grouped, or hybrid computation strategy.

Do not attribute a performance gain to Tensor Core efficiency without confirming supporting profiling evidence.

---

# 9. CORRECTNESS AND TEST REQUIREMENTS

Implement an automated test suite covering:

- Exact output shapes
- Output layout
- Routing indices
- Top-K score propagation
- SwiGLU correctness
- FC1 and FC2 correctness
- Shared-expert contributions
- Non-multiple-of-tile dimensions
- Small token counts
- Large token counts
- Uniform and skewed routing
- Zero routing scores
- Duplicate or invalid IDs, according to contract
- Quantisation scale correctness
- Numerical overflow/underflow
- NaN/Inf behaviour
- Race conditions
- Stream synchronisation
- Deterministic execution where required
- Graph capture compatibility where supported
- Repeated execution without stale workspace corruption

Special attention:

Multiplication by zero does not universally suppress NaN values. Ensure masked experts cannot corrupt results through invalid intermediate values.

Use independent correctness references that do not share the same faulty indexing or layout implementation.

For training-compatible variants, add backward kernels and compare gradients against PyTorch autograd.

Clearly mark inference-only kernels as not supporting backward propagation.

---

# 10. REPRODUCIBLE EXPERIMENTAL METHODOLOGY

All experiments must record:

- Git commit SHA
- GPU model
- Compute capability
- CUDA driver/runtime
- PyTorch version
- Triton version
- CUTLASS version
- TensorRT-LLM version
- CuTe DSL version
- Kernel source hash
- Model dimensions
- Tensor precision
- Weight layout
- Quantisation format
- Routing distribution
- Kernel launch configuration
- Correctness result
- Latency measurements
- Profiling metadata

Measure kernel-only latency separately from end-to-end MoE latency.

Use adequate warm-up, GPU synchronisation, multiple repetitions, and statistical summaries.

Calculate speedup as:

\[
S=\frac{T_{\mathrm{baseline}}}{T_{\mathrm{candidate}}}
\]

Report geometric-mean speedup across explicitly defined workload sets, while preserving individual results to expose regressions.

For significant improvements, repeat the experiment across multiple runs and report uncertainty.

Do not use NVIDIA's published latency numbers as though they were measured on the local hardware.

---

# 11. AGENTIC RL AND SGLANG INTEGRATION

This project may also be used to accelerate LLM policy rollouts.

Investigate whether the optimised MoE kernels can be integrated into inference backends relevant to:

- SGLang
- TensorRT-LLM
- GRPO policy rollout generation
- DAPO/CISPO-related inference workloads

Treat integration with SGLang as a separate implementation milestone.

Do not assume TensorRT-LLM custom kernels can be inserted directly into SGLang without adapter and runtime work.

Evaluate serving-specific conditions:

- Decode-heavy batches
- Prefill workloads
- Continuous batching
- Rapidly changing token counts
- High rollout concurrency
- Variable sequence lengths
- Frequent policy checkpoint updates
- Weight repacking after policy synchronisation

For RL workloads, quantify the impact on:

- Rollout tokens/second
- End-to-end rollout collection time
- GPU memory consumption
- Kernel launch overhead
- Checkpoint/weight update overhead
- Effective GPU utilisation

Distinguish MoE inference kernels from trainable MoE kernels with valid backward implementations.

---

# 12. REQUIRED IMPLEMENTATION DELIVERABLES

Develop or propose the following project structure:

```text
AgenticRL-CUDA-kernels/
└── moe_dense_gemm/
    ├── README.md
    ├── reference/
    │   ├── torch_sparse_moe.py
    │   └── torch_dense_moe.py
    ├── kernels/
    │   ├── cuda/
    │   │   ├── dense_fc1.cu
    │   │   ├── dense_fc2.cu
    │   │   └── alpha_mask.cu
    │   ├── triton/
    │   │   ├── dense_fc1.py
    │   │   ├── dense_fc2.py
    │   │   ├── fused_swiglu.py
    │   │   └── alpha_mask.py
    │   └── cutedsl/
    │       ├── blackwell_fc1.py
    │       └── blackwell_fc2.py
    ├── runtime/
    │   ├── dispatcher.py
    │   ├── weight_packing.py
    │   ├── quantization.py
    │   └── backend_adapter.py
    ├── benchmarks/
    │   ├── bench_fc1.py
    │   ├── bench_fc2.py
    │   ├── bench_full_moe.py
    │   ├── bench_hybrid.py
    │   └── roofline.py
    ├── tests/
    │   ├── test_correctness.py
    │   ├── test_quantization.py
    │   ├── test_routing.py
    │   ├── test_layouts.py
    │   └── test_stream_safety.py
    ├── experiments/
    │   ├── dense_vs_grouped/
    │   ├── active_expert_skip/
    │   ├── router_overlap/
    │   ├── shared_expert_fusion/
    │   └── fc2_alpha_fusion/
    └── results/
```

Reuse existing repository utilities and conventions where appropriate. Do not duplicate functional modules unnecessarily.

For every task, provide:

1. Source implementation
2. PyTorch reference
3. Correctness tests
4. Benchmark script
5. Reproducible execution commands
6. Performance results
7. Profiling interpretation
8. Known limitations
9. Next optimisation hypothesis

---

# 13. RESEARCH-ORIENTED ABLATIONS

Design ablations isolating:

- Dense versus grouped GEMM
- Full dense versus active-expert compaction
- FC1 alpha fusion versus separate scaling
- FC2 alpha fusion versus separate scaling
- Shared-expert fusion versus standalone shared expert
- Router overlap versus serial execution
- Persistent versus nonpersistent scheduling
- Different MMA tile sizes
- Different CTA cluster sizes
- Different weight-packing strategies
- BF16 versus NVFP4
- Different Top-K values
- Different routing entropy
- Different token-batch sizes
- Different tensor-parallel configurations

Do not bundle multiple optimisations into a single comparison without individual ablations.

---

# 14. IMPLEMENTATION PRIORITY AND ACCEPTANCE GATES

Follow this sequence.

### Milestone 1 — Mathematical and Baseline Correctness

- Implement both PyTorch references.
- Test dense/sparse equivalence.
- Establish workload contracts.
- Implement initial routing/alpha generation tests.

Acceptance: all supported correctness tests pass.

### Milestone 2 — Functional GPU Dense GEMM

- Implement BF16/FP16 FC1 and FC2.
- Validate device execution.
- Benchmark against PyTorch.
- Profile major bottlenecks.

Acceptance: genuine GPU kernels execute and pass numerical tests on the target GPU.

### Milestone 3 — Fused MoE Kernels

- Fuse FC1 SwiGLU.
- Fuse FC2 expert alpha.
- Remove avoidable intermediate tensors.
- Confirm end-to-end correctness.

Acceptance: complete routed-MoE output matches the independent reference within defined tolerances.

### Milestone 4 — Blackwell NVFP4

- Implement native FP4 weight/activation handling.
- Add hardware-specific Tensor Core kernels.
- Verify scale transformations.
- Compare against compatible vendor baselines.

Acceptance: reproducible Blackwell execution with numerically validated NVFP4 outputs.

### Milestone 5 — Hybrid Optimisation

- Implement grouped and dense baselines.
- Build a hardware-aware runtime dispatcher.
- Validate crossover predictions.
- Evaluate held-out workloads.

Acceptance: measurable, statistically supported improvements or a clearly documented negative result.

### Milestone 6 — Research Extensions

Investigate:

- Active-expert skipping
- Router-side overlap
- Shared-expert fusion
- Quantisation-aware alpha fusion
- SGLang rollout integration

Acceptance: each experiment has an isolated implementation, complete benchmark methodology, and reproducible results.

---

# 15. FINAL RESEARCH OUTPUTS

Produce:

**A.** Technical architecture report comparing NVIDIA's existing implementation against the proposed kernels.

**B.** Complete source code for each implemented kernel.

**C.** Correctness and numerical-equivalence reports.

**D.** CUDA/Triton/CuTe benchmark results.

**E.** Nsight performance analysis.

**F.** Roofline plots showing memory-bound and compute-bound regimes.

**G.** Dense versus grouped versus hybrid performance curves.

**H.** Detailed ablation-study results.

**I.** Research novelty assessment, identifying which contributions are reproductions, engineering extensions, or potentially novel methods.

**J.** Publication-ready experimental methodology, limitations, and threat-to-validity discussion.

## Final Engineering Rules

- Never report unimplemented kernels as completed.
- Never claim a speedup without direct measured evidence.
- Never confuse arithmetic efficiency with latency efficiency.
- Never substitute a CPU or PyTorch fallback for a custom GPU kernel without identifying it.
- Never assume quantisation-equivalent arithmetic is numerically equivalent.
- Never claim compatibility with unsupported GPU architectures.
- Never treat the NVIDIA reference results as independently reproduced.
- Never omit a slower result from benchmark reporting.
- Always preserve a known-correct baseline before introducing a new optimisation.
- Prefer measured end-to-end improvement over isolated kernel microbenchmark gains.

**Primary objective:** Deliver a reproducible, high-performance, hardware-aware MoE kernel framework that independently establishes when dense GEMM is preferable to grouped GEMM and explores new strategies for reducing low-latency inference overhead on NVIDIA GPUs.
