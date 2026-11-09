"""Trusted, Torch-free workload contracts; candidate workdirs cannot redefine them."""

import os
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Case:
    id: str
    dtype: str
    parameters: tuple

    @property
    def params(self):
        return dict(self.parameters)

    def to_dict(self):
        return {"id": self.id, "dtype": self.dtype, "parameters": self.params}


@dataclass(frozen=True)
class Task:
    name: str
    stage: int
    description: str
    cases: tuple
    sources: tuple
    backends: tuple = ("triton", "cuda")
    eval_only: bool = False
    contract_only: bool = False
    stretch_backends: tuple = ()


SOURCES = {
    "kernelbench": {
        "suite": "KernelBench",
        "url": "https://github.com/ScalingIntelligence/KernelBench",
        "license": "MIT",
        "license_url": "https://github.com/ScalingIntelligence/KernelBench/blob/main/LICENSE",
    },
    "tritonbench": {
        "suite": "Meta-PyTorch TritonBench",
        "url": "https://github.com/meta-pytorch/tritonbench",
        "license": "BSD-3-Clause",
        "license_url": "https://github.com/meta-pytorch/tritonbench/blob/main/LICENSE",
    },
    "tutorials": {
        "suite": "Official Triton tutorials",
        "url": "https://triton-lang.org/main/getting-started/tutorials/index.html",
        "license": "MIT",
        "license_url": "https://github.com/triton-lang/triton/blob/main/LICENSE",
    },
    "gpu_mode": {
        "suite": "GPU MODE reference-kernels (evaluation-only adaptation)",
        "url": "https://github.com/gpu-mode/reference-kernels",
        "license": "June 9 Researcher Reciprocity License 1.0",
        "license_url": "https://github.com/gpu-mode/reference-kernels/blob/main/LICENSE",
    },
    "flashinfer": {
        "suite": "FlashInfer-Bench",
        "url": "https://github.com/flashinfer-ai/flashinfer-bench",
        "license": "Apache-2.0",
        "license_url": "https://github.com/flashinfer-ai/flashinfer-bench/blob/main/LICENCE",
    },
    "local": {
        "suite": "Original AgenticRL workload derived from local GRPO/CISPO configuration",
        "url": "https://github.com/ElenaViewSynthesis/RL-Army-of-Agents",
        "license": "NOASSERTION: repository has no root license file",
        "license_url": None,
    },
    "dense_moe": {
        "suite": "TensorRT-LLM MoE as Dense GEMM (BF16 H200 adaptation)",
        "url": "https://nvidia.github.io/TensorRT-LLM/1.3.0rc15/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html",
        "license": "Apache-2.0 (TensorRT-LLM project); article attribution retained",
        "license_url": "https://github.com/NVIDIA/TensorRT-LLM/blob/main/LICENSE",
    },
}


def case(dtype="float32", **parameters):
    suffix = "_".join(f"{k}{v}".lower().replace(".", "p") for k, v in parameters.items())
    return Case(f"{suffix}_{dtype}" if suffix else dtype, dtype, tuple(parameters.items()))


TASKS = {
    "gemm_div_sum_scale": Task(
        "gemm_div_sum_scale", 0,
        "scale * sum((A.float() @ B.float()) / divisor, dim=1, keepdim=True); output fp32",
        tuple(case(dtype, m=m, n=n, k=k) for dtype in ("float32", "bfloat16")
              for m, n, k in product((256, 512, 1024), repeat=3)),
        ("kernelbench", "tutorials")),
    "row_softmax": Task(
        "row_softmax", 1, "Stable row softmax with fp32 accumulation; output input dtype",
        tuple(case(dtype, b=b, h=h) for dtype in ("float32", "float16", "bfloat16")
              for b, h in product((64, 512), (128, 1024, 8192))),
        ("kernelbench", "tutorials")),
    "masked_mean": Task(
        "masked_mean", 1,
        "Mean of x[row,:lengths[row]]; empty rows return fp32 zero; this is not trimmed mean",
        tuple(case(dtype, b=b, l=l) for dtype in ("float32", "bfloat16")
              for b, l in product((64, 512), (128, 256, 512, 1024, 2048, 4096, 8192))),
        ("gpu_mode",), eval_only=True),
    "prefix_sum": Task(
        "prefix_sum", 1,
        "Global inclusive scan, not independent block scans; nonnegative inputs; int32 does not overflow",
        tuple(case(dtype, n=n) for dtype in ("int32", "float32")
              for n in (100000, 1000000, 10000000)),
        ("gpu_mode",), eval_only=True),
    "gemm_tensor_core": Task(
        "gemm_tensor_core", 2,
        "A @ B + bias[N], fp16/bf16 operands with fp32 accumulation and fp32 output",
        tuple(case(dtype, m=s, n=s, k=s) for dtype in ("bfloat16", "float16")
              for s in (512, 1024, 2048, 4096))
        + tuple(case(dtype, m=m, n=4096, k=4096)
                for dtype in ("bfloat16", "float16") for m in (8, 64)),
        ("tritonbench", "tutorials"), backends=("triton", "cuda", "cutedsl")),
    "rmsnorm": Task(
        "rmsnorm", 2,
        "x * rsqrt(mean(x.float()**2)+eps) * weight[H]; output input dtype",
        tuple(case(dtype, b=b, h=h) for dtype in ("bfloat16", "float32")
              for b, h in product((64, 512), (1024, 2048, 4096, 8192))),
        ("tritonbench", "tutorials")),
    "log_softmax_gather": Task(
        "log_softmax_gather", 2,
        "Stable fp32 log-softmax followed by one token-index gather per row; fp32 output",
        tuple(case(dtype, b=b, v=v) for dtype in ("bfloat16", "float32")
              for b, v in product((32, 256), (1024, 8192, 128256))),
        ("local",)),
    "fused_adamw": Task(
        "fused_adamw", 2,
        "Functional bias-corrected AdamW: return (new_parameter,new_m,new_v), never mutate inputs",
        tuple(case(n=n) for n in (4096, 1048576, 8388608)),
        ("local",)),
    "layernorm_residual_gelu": Task(
        "layernorm_residual_gelu", 2,
        "Affine population LayerNorm, residual add, exact erf GELU; fp32 arithmetic then output cast",
        tuple(case(dtype, b=b, h=h) for dtype in ("float32", "bfloat16")
              for b, h in product((64, 512), (1024, 4096, 8192))),
        ("kernelbench", "tutorials")),
    "grpo_group_norm": Task(
        "grpo_group_norm", 3,
        "(reward-population_mean)/(population_std+1e-6), matching utils.grpo.group_advantages",
        tuple(case(b=b, g=g) for b, g in product((64, 512), (1, 4, 8, 16))),
        ("local",)),
    "cispo_clip_weight": Task(
        "cispo_clip_weight", 3,
        "mask * clip(exp(log_ratio),0,5).detach() * sequence_advantage; fp32 forward weights",
        tuple(case(b=b, s=s) for b, s in product((8, 64), (128, 1024, 8192))),
        ("local",)),
    "packed_varlen_reduction": Task(
        "packed_varlen_reduction", 3,
        "Per-segment fp32 mean or sum from packed values and monotone offsets; empty segments return zero",
        tuple(case(b=b, l=l, reduction=reduction)
              for b, l, reduction in product((64, 512), (128, 2048, 8192), ("mean", "sum"))),
        ("local",)),
    "flash_attention_fwd": Task(
        "flash_attention_fwd", 4,
        "Causal softmax(Q @ K.T / sqrt(64)) @ V; fp32 attention arithmetic, bf16 output; tolerance 1e-2",
        tuple(case("bfloat16", b=4, h=8, s=s, d=64) for s in (512, 1024, 2048)),
        ("kernelbench", "tutorials"), stretch_backends=("cutedsl", "tilelang")),
    "grouped_gemm": Task(
        "grouped_gemm", 4,
        "List of independent bf16 A_i @ B_i, variable per-group M, fp32 outputs",
        tuple(case("bfloat16", g=g, n=n, k=n) for g, n in product((4, 8), (512, 1024))),
        ("tritonbench", "tutorials"), backends=("triton",)),
    "topk_sampling": Task(
        "topk_sampling", 4,
        "Stable top-k-first, then nucleus filtering, then inverse-CDF sampling using explicit uniforms",
        tuple(case(b=b, v=128256, top_k=k, top_p=p)
              for b, k, p in product((1, 8, 32), (1, 16, 32), (0.9, 1.0))),
        ("flashinfer",)),
    "residual_rmsnorm": Task(
        "residual_rmsnorm", 4,
        "BF16 residual sum, fp32 RMSNorm accumulation; return (normalized,residual_sum), both bf16",
        tuple(case("bfloat16", b=b, s=s, h=h)
              for b, s, h in product((1, 4), (128, 512, 2048), (1024, 4096, 8192))),
        ("flashinfer",)),
    "decode_attention": Task(
        "decode_attention", 4,
        "Single last-position query, 32 query heads / 8 KV heads, d=64; bf16 inputs/output",
        tuple(case("bfloat16", b=b, hq=32, hkv=8, s=s, d=64)
              for b, s in product((1, 4), (512, 2048, 8192))),
        ("flashinfer", "tutorials")),
    "moe_dense_gemm": Task(
        "moe_dense_gemm", 4,
        "Dense all-expert FC1, fp32 SwiGLU and alpha mask, bf16 intermediate, dense FC2; fp32 output",
        tuple(case("bfloat16", t=t, h=h, i=i, e=e, top_k=2, shared=shared)
              for t, h, i, e, shared in (
                  (8, 256, 512, 4, False), (32, 256, 512, 4, False),
                  (64, 256, 512, 4, False), (128, 256, 512, 4, False),
                  (64, 512, 512, 8, False), (64, 512, 512, 8, True))),
        ("dense_moe", "tutorials")),
    "fp8_gemm": Task(
        "fp8_gemm", 4,
        "Contract only: E4M3FN operands, explicit scales, fp32 accumulation/output; no measured fixture",
        tuple(case("float8_e4m3fn", m=s, n=s, k=s) for s in (256, 512, 1024)),
        ("gpu_mode",), backends=(), eval_only=True, contract_only=True),
    "int8_gemm": Task(
        "int8_gemm", 4,
        "Contract only: int8 operands, int32 accumulation, explicit row/column scales, fp32 output",
        tuple(case("int8", m=s, n=s, k=s) for s in (256, 512, 1024)),
        ("gpu_mode",), backends=(), eval_only=True, contract_only=True),
}


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"Unknown suite task: {name}")
    return TASKS[name]


def get_case(task_name, case_id=None):
    task = get_task(task_name)
    if case_id is None:
        return task.cases[0]
    for item in task.cases:
        if item.id == case_id:
            return item
    raise ValueError(f"Unknown case {case_id!r} for {task_name}")


def active_case(task_name):
    declared = os.environ.get("CUDA_AGENT_TASK")
    if declared is not None and declared != task_name:
        raise ValueError("Workdir reference does not match the selected trusted task")
    return get_case(task_name, os.environ.get("CUDA_AGENT_TASK_CASE"))


def select_cases(task_name, selection="smoke"):
    task = get_task(task_name)
    if selection == "all":
        return task.cases
    if selection == "smoke":
        return task.cases[:1]
    if selection == "acceptance":
        if task_name == "gemm_tensor_core":
            return (get_case(task_name, "m4096_n4096_k4096_bfloat16"),)
        if task_name == "flash_attention_fwd":
            return task.cases[-1:]
        return task.cases[:1]
    return (get_case(task_name, selection),)


def validate_usage(task_name, purpose):
    task = get_task(task_name)
    if purpose not in ("eval", "train"):
        raise ValueError("purpose must be eval or train")
    if purpose == "train" and task.eval_only:
        raise ValueError(f"{task_name} is eval-only; GPU MODE training reciprocity is not cleared")


def workload(task_name, item):
    p = item.params
    size = {"float32": 4, "float16": 2, "bfloat16": 2, "int32": 4,
            "int8": 1, "float8_e4m3fn": 1}[item.dtype]
    flops, logical_bytes, convention = None, None, "unavailable"
    if task_name in ("gemm_div_sum_scale", "gemm_tensor_core", "fp8_gemm", "int8_gemm"):
        m, n, k = p["m"], p["n"], p["k"]
        flops, convention = 2 * m * n * k, "GEMM 2MNK; epilogue excluded"
        output = m if task_name == "gemm_div_sum_scale" else m * n
        logical_bytes = (m * k + k * n) * size + output * 4
        if task_name == "gemm_tensor_core":
            logical_bytes += n * 4
    elif task_name == "flash_attention_fwd":
        b, h, s, d = (p[k] for k in ("b", "h", "s", "d"))
        flops = 2 * b * h * s * (s + 1) * d
        logical_bytes, convention = 4 * b * h * s * d * size, "algorithmic causal QK+PV; triangular"
    elif task_name == "decode_attention":
        b, hq, hkv, s, d = (p[k] for k in ("b", "hq", "hkv", "s", "d"))
        flops = 4 * b * hq * s * d
        logical_bytes = (2 * b * hq * d + 2 * b * hkv * s * d) * size
        convention = "algorithmic single-query QK+PV"
    elif task_name == "moe_dense_gemm":
        t, h, i = p["t"], p["h"], p["i"]
        experts = p["e"] + int(p["shared"])
        flops, convention = 6 * t * experts * h * i, "all-expert dense GEMMs; SwiGLU/mask excluded"
        logical_bytes = (t * h + 3 * experts * h * i) * size + t * h * 4 + t * experts * 4
    elif task_name in ("row_softmax", "rmsnorm", "layernorm_residual_gelu"):
        count = p["b"] * p["h"]
        logical_bytes = 2 * count * size
        if task_name == "rmsnorm":
            logical_bytes += p["h"] * size
        if task_name == "layernorm_residual_gelu":
            logical_bytes += (count + 2 * p["h"]) * size
    elif task_name == "prefix_sum":
        logical_bytes = 2 * p["n"] * size
    elif task_name == "fused_adamw":
        logical_bytes = 7 * p["n"] * 4
    return {"flops": flops, "logical_io_bytes": logical_bytes, "flop_convention": convention,
            "bandwidth_convention": "logical input/output bytes, not measured HBM traffic"}
