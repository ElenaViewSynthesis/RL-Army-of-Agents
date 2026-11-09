import hashlib
import json
import os
import sys
from pathlib import Path


source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
repo = Path("/mnt/c/Users/proxi/Documents/codex-9/CUDA-RL/AgenticRL-CUDA-kernels")
stamp = source.name.removeprefix("task-suite-acceptance-source-")
manifest_bytes = (source / "task-source-manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
for name, digest in manifest["files"].items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Frozen source mismatch: {name}")
if os.environ.get("MODAL_IMAGE_BUILDER_VERSION") != "2025.06":
    raise ValueError("This plan requires the unchanged 2025.06 builder")

import modal
from cloud.modal_job import wait_for_call
from cloud.modal_tasks import app, evaluate
from utils.task_catalog import select_cases


plan = [
    ("cute-retest", "gemm_tensor_core", "smoke", "cutedsl"),
    ("flagship-acceptance", "gemm_tensor_core,flash_attention_fwd", "acceptance", "cuda,triton"),
    ("flash-middle", "flash_attention_fwd", "b4_h8_s1024_d64_bfloat16", "cuda,triton"),
    ("grpo-nontrivial", "grpo_group_norm", "b512_g16_float32", "cuda,triton"),
    ("sampling-k32", "topk_sampling", "b32_v128256_top_k32_top_p0p9_float32", "cuda,triton"),
    ("grouped-large", "grouped_gemm", "g8_n1024_k1024_bfloat16", "triton"),
    ("moe-shared", "moe_dense_gemm", "t64_h512_i512_e8_top_k2_sharedtrue_bfloat16", "cuda,triton"),
    ("residual-large", "residual_rmsnorm", "b4_s2048_h8192_bfloat16", "cuda,triton"),
    ("decode-long", "decode_attention", "b4_hq32_hkv8_s8192_d64_bfloat16", "cuda,triton"),
    ("scan-large-exact", "prefix_sum", "n10000000_int32", "cuda,triton"),
    ("layernorm-bf16", "layernorm_residual_gelu", "b512_h8192_bfloat16", "cuda,triton"),
    ("packed-sum", "packed_varlen_reduction", "b512_l8192_reductionsum_float32", "cuda,triton"),
]
for _, tasks, selection, _ in plan:
    for task in tasks.split(","):
        if task not in manifest["tasks"]:
            raise ValueError(f"Task missing from frozen source: {task}")
        select_cases(task, selection)

previous = wait_for_call(modal.FunctionCall.from_id("fc-01M6Z0JEB49RQ3G743941CP86E").hydrate())
print(json.dumps({"previous_advanced_job": previous}), flush=True)
if previous.get("status") == "incomplete":
    raise RuntimeError("Previous job incomplete; stop before scheduling more GPU work")
with app.run(detach=True):
    for suffix, tasks, selection, backends in plan:
        run_id = f"task-{suffix}-{stamp}"
        local = repo / "artifacts" / run_id
        local.mkdir(exist_ok=False)
        call = evaluate.spawn(run_id, tasks, selection, backends, "honest,hack,wrong",
                              600, True, True)
        receipt = {
            "run_id": run_id, "function_call_id": call.object_id, "source": str(source),
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_files": len(manifest["files"]), "builder_requested": "2025.06",
            "workspace_default_changed": False, "tasks": tasks, "case_set": selection,
            "backends": backends, "fixtures": "honest,hack,wrong",
            "benchmark": True, "cold_start": True, "volume": "agenticrl-cuda-artifacts",
            "remote": f"/artifacts/{run_id}",
        }
        with (local / "launch-receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"spawned": receipt}), flush=True)
        summary = wait_for_call(call)
        with (local / "durable-result.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"completed": summary}), flush=True)
        if summary.get("status") == "incomplete" or summary.get("source_mismatches"):
            raise RuntimeError("Incomplete execution or source mismatch; remaining GPU plan stopped")
