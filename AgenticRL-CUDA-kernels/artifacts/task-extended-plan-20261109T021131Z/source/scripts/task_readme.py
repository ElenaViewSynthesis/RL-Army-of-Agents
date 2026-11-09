"""Render task READMEs from trusted contracts and already-frozen result records."""

import argparse
import json
import math
from pathlib import Path

from utils.task_catalog import SOURCES, TASKS, get_task


ROOT = Path(__file__).resolve().parent.parent
DETAILS = {
    "gemm_div_sum_scale": (
        "This adapts KernelBench level 2 problem 14, "
        "`KernelBench/level2/14_Gemm_Divide_Sum_Scaling.py`. Upstream stores the right-hand "
        "weight as a module parameter; this functional task takes `A,B` as inputs instead. "
        "The constructor is `(divisor=2.0, scale=1.5)`, its state_dict is empty, and the "
        "output is `(M,1)` fp32. There is no bias. Sum over N, not M. FP32 reference "
        "matmul uses highest precision. Honest fixtures perform a real GEMM then reduction; "
        "the reported GEMM FLOP convention excludes the reduction epilogue."),
    "masked_mean": (
        "Lengths are int32 and satisfy `0 <= length <= L`. At least one generated row is "
        "empty and one is full. Empty rows return zero. This is a masked/prefix mean, "
        "not an order-statistic trimmed mean. It is a modified problem-family adaptation, "
        "not a claim to reproduce a particular GPU MODE competition submission."),
    "prefix_sum": (
        "The scan is globally inclusive across block boundaries. Integer inputs are in "
        "`[0,3]`, so the declared largest shape cannot overflow int32. Floating inputs are "
        "nonnegative fp32 in `[0,1)`. The multi-pass implementation recursively scans block "
        "totals and then adds offsets; independent block scans are incorrect."),
    "gemm_tensor_core": (
        "Inputs are `A:(M,K)`, `B:(K,N)`, and fp32 `bias:(N,)`; output is fp32, not a "
        "bf16-rounded GEMM converted to fp32 afterward. Triton retains bf16/fp16 dot operands "
        "and accumulates in fp32. The CUDA fixture uses explicit WMMA/mma Tensor Core work "
        "with double-buffered shared-memory tiles. CuTe DSL must use MMA/copy atoms, not "
        "CUTLASS utility GEMM templates. The honest fixture acceptance target is Triton "
        "**at least 60% of the recorded cuBLAS throughput at bf16 4096^3 on H200**, "
        "with authored non-memory provenance **exactly 1.0**. This is unaccepted until "
        "measured. Skinny M=8 and M=64 cases are included. Candidates may not call cuBLAS."),
    "fused_adamw": (
        "Forward returns `(updated_parameter,new_m,new_v)` without mutating any input. "
        "The constructor supplies learning rate, beta1/beta2, epsilon, weight decay and "
        "step; bias correction is explicit. This is a functional numerical kernel "
        "benchmark, not a trainer or an integrated optimizer."),
    "layernorm_residual_gelu": (
        "All normalization/affine/residual/GELU arithmetic is fp32 before the final output "
        "cast. Variance is population variance of centered values. GELU uses erf, not "
        "the tanh approximation. This is an adapted fused-operator workload, not an "
        "upstream harness port."),
    "grpo_group_norm": (
        "The reference follows `utils/grpo.py:group_advantages`: population standard "
        "deviation and denominator `std + eps`, **not** `sqrt(variance + eps)`, with "
        "`eps=1e-6`. Singleton and constant-reward groups return zero. Group filtering "
        "and task grouping remain host-side; this kernel does not implement a trainer."),
    "cispo_clip_weight": (
        "`get_init_inputs()` derives `[1-clip_low,1+clip_high]` from "
        "`RolloutConfig.preset('cispo')` in `utils/config.py`, currently `[0,5]`. Inputs "
        "are fp32 `log_ratio:(B,S)`, sequence `advantage:(B,)`, and boolean mask `(B,S)`. "
        "The forward result is masked clipped IS weight times advantage. The detached "
        "IS-weight rule is explicit in the reference; this is a forward-only contract, "
        "not a claim of custom autograd, token-loss backpropagation or trainer integration."),
    "packed_varlen_reduction": (
        "Inputs are packed fp32 values and monotone int32 offsets of length B+1, starting "
        "at zero and ending at the packed length. Consecutive equal offsets represent "
        "empty sequences. The constructor selects mean or sum; both return zero for "
        "empty segments. This is the reduction primitive for sequence/token aggregation "
        "motivated by `RolloutConfig.loss_agg`, not a replacement for the host GRPO batch builder."),
    "flash_attention_fwd": (
        "The constructor is `(causal=True)`. `get_inputs()` returns `(Q,K,V)` with "
        "`B=4,H=8,d=64,S in {512,1024,2048}` and bf16 inputs/output. Reference QK, softmax "
        "and PV arithmetic is fp32. Tolerance stays **atol=rtol=1e-2**; 2e-2 was not "
        "authorized. The honest Triton path uses tiled online softmax; CUDA uses a "
        "two-pass shared-memory implementation. CuTe DSL MMA/online-softmax and TileLang "
        "T.gemm/pipelined-softmax are stretch backends, not implicitly passed fixtures. "
        "The deliberate reward hack calls PyTorch scaled_dot_product_attention. "
        "Algorithmic causal FLOPs are `2*B*H*S*(S+1)*d`, approximately half the full "
        "`4*B*H*S*S*d`; these are not measured executed-instruction counts. A baseline "
        "must identify its actual Flash/SDPA provider/version before a ceiling percentage "
        "is reported; an unverified provider must not be relabeled FA2."),
    "grouped_gemm": (
        "Triton-only initially, as requested. Inputs and outputs are Python lists of "
        "matrices. Groups use M cycling through `{8,32,64,128}`, with per-case common N/K. "
        "There are no padded rows in the reference output. Each output is fp32."),
    "topk_sampling": (
        "The constructor supplies top_k/top_p. Inputs are fp32 logits `(B,128256)` and "
        "explicit fp32 uniforms `(B,)` in `[0,1)`. Sort descending with lower token index "
        "breaking ties. Apply top-k first, softmax over the kept logits, keep the smallest "
        "prefix reaching top_p, renormalize, and choose the first CDF strictly greater "
        "than the uniform. Output token IDs are int32. Explicit uniforms make verification "
        "deterministic; candidates must not generate unrelated RNG samples. This adapts "
        "FlashInfer's top_k_first semantics, not its rejection-sampling harness."),
    "residual_rmsnorm": (
        "Flatten prefill tokens to `(B*S,H)`. Round the residual sum to bf16 **before** "
        "the fp32 RMS reduction, matching the reference. Return `(normalized,residual_sum)`, "
        "both bf16. Inputs are not mutated."),
    "decode_attention": (
        "Q is `(B,32,1,64)`, K/V are `(B,8,S_kv,64)`. Each four consecutive query heads "
        "share a KV head. The single query is at the final cache position and sees the "
        "entire cache. Accumulation/softmax is fp32, output bf16."),
    "fp8_gemm": (
        "**Contract only: no authored GPU fixture or timing baseline is shipped yet.** "
        "Operands are E4M3FN tensors, with explicit fp32 row scales for A and column scales "
        "for B. Reference output is `(A.float()@B.float())*scale_a[:,None]*scale_b[None,:]`. "
        "Scales are inputs, not inferred from output. No tolerance relaxation is approved."),
    "int8_gemm": (
        "**Contract only: no authored GPU fixture or timing baseline is shipped yet.** "
        "Operands are signed int8; products accumulate in int32 without overflow for "
        "the declared generated range. Convert the accumulator to fp32, then multiply "
        "explicit row/column scales. No library-kernel candidate is permitted."),
}


DENSE_DETAILS = """
### Dense-MoE BF16 adaptation

This is a methodology adaptation of the linked Blackwell article, not a port of
TensorRT-LLM's prewritten kernels. With E routed experts and optional one shared
expert, inputs are X `(T,H)`, gate/up weights `(E_total,H,I)`, down weights
`(E_total,I,H)`, and fp32 alpha `(T,E_total)`. Alpha is zero for unselected routed
experts, normalized positive weights for selected experts, and one for the optional
shared expert. All expert FC1 columns and FC2 K blocks are computed densely:

`FC1 = X @ concat(gate,up)` with fp32 accumulation; then
`A = (SiLU(gate_result) * up_result * alpha).to(bf16)`; then
`O = A.flatten(experts,I) @ down.flatten(experts,I)` with fp32 accumulation/output.

Intermediate I is a multiple of 256. This implementation's logical gate/up packing
is explicit in model.py; the article's 64-column interleave and NVFP4 scale swizzles
are not claimed to be the H200 layout. Dense-GEMM FLOP/s counts all experts, not just
the routed top-k. Do not compare those counts with useful-routed FLOPs as if they
were the same work. No Blackwell speedup from the article is a repository baseline.

### NVFP4/Blackwell contract (not implemented on H200)

This separate future precision contract requires Blackwell NVFP4 MMA support.
Shapes and routing math follow the BF16 contract, with I 256-aligned. Logical
FP4 values are E2M1 with magnitudes `{0,0.5,1,1.5,2,3,4,6}`, two signed nibbles
per byte, increasing logical index in the low nibble first. Each contiguous
16-value K block has a positive E4M3FN scale and each tensor a positive fp32
global scale; the logical dequantization is
`decode_fp4(nibble) * decode_e4m3(block_scale) * global_scale`.

The quantization-aware reference must dequantize X and all weights, perform
dense FC1 in fp32, apply SwiGLU, requantize the activation in the same block
format using round-to-nearest-even with saturation to magnitude 6, then
dequantize it for FC2. Activation block scales are explicit inputs to this
contract, not a hidden calibration algorithm. Routing alpha is applied per
token/expert before the corresponding FC2 K block; unselected experts have
alpha zero. W2 logical storage is hidden-major `(H,E_total*I)`; physical scale
swizzles must be separately specified by a future hardware implementation.

There is no NVFP4 candidate, performance number, native H200 support, or accepted
precision-tolerance claim. Existing 1e-2 tolerance is not silently relaxed.
"""


def _measured(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _display(value):
    return f"{value:.6f}" if _measured(value) else "unavailable"


def render(task_name, records=(), evidence=None, limitations=()):
    task = get_task(task_name)
    sources = []
    for key in task.sources:
        source = SOURCES[key]
        license_text = f"[{source['license']}]({source['license_url']})" if source["license_url"] else source["license"]
        sources.append(f"- {source['suite']}: {source['url']}; license: {license_text}.")
    rows = "\n".join(f"| `{item.id}` | `{item.dtype}` | `{item.params}` |" for item in task.cases)
    note = DETAILS.get(task_name, "")
    if task_name == "moe_dense_gemm":
        note = DENSE_DETAILS
    license_note = (
        "\n**Held-out evaluation only.** GPU MODE's June 9 Researcher Reciprocity License "
        "covers evaluation/benchmarking for training as Training Use. This modified "
        "problem-family adaptation is blocked from `purpose=train` until reciprocity is "
        "cleared. Labeling an RL feedback run 'eval' does not clear those obligations.\n"
        if task.eval_only else "")
    baselines, ceilings = [], []
    for record in records:
        if (record.get("suite/task") != task_name or record.get("suite/fixture") != "honest"
                or record.get("suite/status") != "passed" or not record.get("bench/available")):
            continue
        values = [record.get(f"bench/{name}/event_us_median") for name in ("eager", "compile", "generated")]
        if not all(_measured(value) for value in values):
            continue
        baselines.append(f"| `{record['suite/case']}` | `{record['suite/backend']}` | "
                         + " | ".join(_display(value) for value in (
                             *values, record.get("bench/generated/cold_event_us"),
                             record.get("bench/generated/tflops"),
                             record.get("bench/generated/logical_gbytes_per_s"))) + " |")
        keys = ("ceiling/cublas_event_us_median", "ceiling/generated_event_us_median",
                "ceiling/cublas_tflops", "ceiling/generated_tflops", "ceiling/cublas_fraction")
        vendor_values = [record.get(key) for key in keys]
        if record.get("ceiling/available") is True and all(_measured(value) for value in vendor_values):
            ceilings.append(f"| `{record['suite/case']}` | `{record['suite/backend']}` | "
                            f"`{record.get('ceiling/provider', 'unrecorded')}` | "
                            + " | ".join(_display(value) for value in vendor_values[:-1])
                            + f" | {vendor_values[-1]:.2%} |")
    measured = "Unavailable: no measured GPU baseline has been recorded here."
    if baselines:
        if not evidence or not evidence.get("run_id") or not evidence.get("command"):
            raise ValueError("Measured README baselines require their captured run ID and exact command")
        runtime = evidence.get("runtime", {})
        identity = ", ".join(str(runtime.get(key, "unrecorded")) for key in (
            "runtime/gpu0_name", "runtime/gpu0_compute_cap", "runtime/gpu0_uuid",
            "runtime/torch_version", "runtime/triton_version"))
        measured = (
            f"Measured run: `{evidence['run_id']}`. Runtime (GPU, capability, UUID, Torch, Triton): "
            f"`{identity}`. Frozen source: `{evidence.get('source_snapshot', 'unrecorded')}`.\n\n"
            "Recorded command (use a **new** output directory when reproducing):\n\n"
            f"```bash\n{evidence['command']}\n```\n\n"
            "Warm columns are CUDA-event medians, not reward-profiler sums. Cold is a "
            "single first-forward sample after initialization, not a cold-cache median. "
            "Rates are recorded workload diagnostics; logical GB/s is not HBM traffic.\n\n"
            "| Case | Backend | Eager warm us | Compile warm us | Generated warm us | "
            "Generated first-forward us | Generated TFLOP/s | Generated logical GB/s |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(baselines))
        if ceilings:
            measured += (
                "\n\nSeparately measured **cuBLAS GEMM-only with fp32 output**; generated "
                "includes bias. These paired samples are distinct from the warm table above. "
                "This is neither a CUTLASS-profiler nor a FlashAttention baseline.\n\n"
                "| Case | Backend | Provider | cuBLAS us | Generated+bias us | cuBLAS TFLOP/s | "
                "Generated TFLOP/s | Percent of cuBLAS |\n"
                "|---|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(ceilings))
        if limitations:
            measured += "\n\nCaptured measurement limits:\n" + "\n".join(f"- {item}" for item in limitations)
    return f"""# `{task.name}`

## Operator contract

{task.description}

{note}

The reference constructor and state_dict must match ModelNew. These functional
tasks take weights/state arrays as inputs; state_dict is empty. Candidate wrappers
may allocate tensors and inspect metadata, but must not perform Torch computation.
All device work must join `torch.cuda.current_stream()` before returning.

## Source provenance and licensing

{chr(10).join(sources)}
{license_note}
Definitions, grids, layouts and implementations are adaptations, not copied
upstream harnesses or claims of identical upstream problem IDs. Tutorial material
is an implementation reference, not permission to call prewritten library kernels.
The original repository has no root LICENSE file; external source licenses above
are not a new blanket license for this repository.

## Shape/dtype grid

| Case | Input dtype | Parameters |
|---|---|---|
{rows}

## Fixtures and gates

- Honest backends: {", ".join(task.backends) or "none (contract only)"}.
- Stretch backends: {", ".join(task.stretch_backends) or "none declared"}.
- `model_new.py` selects the default honest fixture. `model_new_hack.py` is an
  intentional Torch reference-compute attack; it must verify then fail provenance.
- `model_new_wrong.py` intentionally returns NaNs or invalid integer sentinels;
  it must fail verification, not merely compilation.
- Verification uses five fresh random inputs; floating outputs keep atol=rtol=1e-2.
  The progressive suite requires exact equality for integer outputs, including
  token IDs and prefix sums. `utils.task_verification` exposes the same strict
  check directly; stock `utils.verification` and AXPBY defaults stay unchanged.
- The suite alone applies authored-only provenance: at least 90% of non-memory
  device time, no library-kernel time, and 100% authored for gemm_tensor_core.
  Stock AXPBY attribution/reward rules are unchanged.
- No triton.ops, flash-attn packages, cuBLAS/cuDNN/CUTLASS-library candidate calls.
  tl.dot and authored CuTe MMA/copy atoms are allowed.
- Name matching remains a heuristic, not proof of authorship or a security sandbox.
- CPU/missing capabilities are **unavailable**, never GPU passes.

## Reproduction and measured baselines

From the project root, on an H200 runtime:

```bash
python -m scripts.validate --suite {task.name} --device cuda --purpose eval \\
  --case-set smoke --benchmark --cold-start --timeout-s 600
```

Use `--case-set all` for the declared grid, or an exact case ID. Output is in a
new artifacts/task-suite-* directory; progress.jsonl retains completed cells.
`--case-set acceptance` selects 4096^3 bf16 GEMM or S=2048 attention where applicable.
Warm event/wall samples and the separately defined first-forward sample are
diagnostics, never reward timings. The first-forward sample is not cache-flushed
and may include lazy compilation. Logical bandwidth is not measured HBM traffic.
cuBLAS, CUTLASS-profiler and Flash-provider baselines must retain their identities;
no CUTLASS result is relabeled cuBLAS and no unverified Flash provider is called FA2.

{measured}

No absent timing, TFLOP/s, bandwidth, ceiling percentage, or H200 acceptance is zero-filled.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--write-new", action="store_true", help="Create READMEs; refuse to overwrite")
    args = parser.parse_args()
    names = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    capture = json.loads(args.records.read_text()) if args.records else {}
    records = capture.get("rows", ())
    for name in names:
        text = render(name, records, capture.get("evidence"), capture.get("limitations", ()))
        if args.write_new:
            with (ROOT / "tasks" / name / "README.md").open("x") as stream:
                stream.write(text)
        else:
            print(text)


if __name__ == "__main__":
    main()
