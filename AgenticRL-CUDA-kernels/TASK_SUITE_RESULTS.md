# Task suite: results worth knowing

- The Triton tensor-core GEMM runs at 85.4% of cuBLAS on a 4096³ bf16
  multiply, well above the 60% gate. The fixed CUDA version also passes.
- Every stage 4 task with an implemented backend passes on the H200, even though most of them failed or
  didn't finish on the CPU interpreter. Those CPU failures were interpreter
  limits, not kernel bugs.
- The CuTe DSL tensor-core GEMM now compiles and passes on the H200 (smoke
  and 4096³ acceptance, 3/3 each), but it is slow: 2.6% of cuBLAS at 4096³.

## Evidence

| Result | H200 run |
|---|---|
| Triton GEMM at 0.854 of cuBLAS (`gemm_tensor_core`, 4096³ bf16; gate ≥ 0.6); CUDA GEMM passes after its build fix | `task-flagship-acceptance-20261109T021131Z` |
| Stage 4 smoke cases: 33 passed, 3 unavailable (`grouped_gemm` has no CUDA fixture by design) | `task-stage4-smoke-20261109T020100Z` |
| Stage 4 larger cases (decode and flash attention, grouped GEMM, Dense-MoE, residual RMSNorm, top-k sampling): all pass | `task-*-20261109T021131Z` |
| CuTe DSL GEMM after the C-store fix: 512³ and 4096³ bf16 pass 3/3; honest kernel at 0.143 and 0.026 of cuBLAS | `task-cute-fix-{smoke,acceptance}-20261109T030234Z` |

The CPU results for the same tasks are in `artifacts/task-stage4-20261109T005506Z/`.
Under the interpreter, the bf16 `tl.dot` kernels mismatched, `residual_rmsnorm`
had 35 of 131072 elements out of tolerance, and `topk_sampling` did not
finish. Scope: one H200 and the listed cases only. Kernel attribution is
name-based. See [README.md](README.md#kernel-tasks-the-agent-works-on),
[VALIDATION.md](VALIDATION.md) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the
full tables and open items.
