"""Input generation for the trusted task catalog; Torch is imported only on use."""

from utils.task_catalog import active_case


def make_inputs(task_name):
    import torch

    item = active_case(task_name)
    if task_name.startswith("eagle_"):
        from utils.eagle_reference import make_inputs as eagle_inputs
        return eagle_inputs(task_name, item)
    if task_name.startswith("lora_"):
        from utils.lora_reference import make_inputs as lora_inputs
        return lora_inputs(task_name, item)
    p = item.params
    dtype = getattr(torch, item.dtype)
    randn = lambda *shape: torch.randn(*shape, dtype=dtype)
    if task_name in ("gemm_div_sum_scale", "gemm_tensor_core"):
        result = [randn(p["m"], p["k"]), randn(p["k"], p["n"])]
        if task_name == "gemm_tensor_core":
            result.append(torch.randn(p["n"], dtype=torch.float32))
        return result
    if task_name == "row_softmax":
        return [randn(p["b"], p["h"])]
    if task_name == "masked_mean":
        lengths = torch.randint(0, p["l"] + 1, (p["b"],), dtype=torch.int32)
        lengths[0], lengths[1] = 0, p["l"]
        return [randn(p["b"], p["l"]), lengths]
    if task_name == "prefix_sum":
        if dtype == torch.int32:
            return [torch.randint(0, 4, (p["n"],), dtype=dtype)]
        return [torch.rand(p["n"], dtype=dtype)]
    if task_name == "rmsnorm":
        return [randn(p["b"], p["h"]), randn(p["h"])]
    if task_name == "relu_squared":
        return [randn(p["b"], p["h"])]
    if task_name in ("silu_and_mul", "geglu_tanh"):
        return [randn(p["b"], 2 * p["h"])]
    if task_name == "rotary_embedding":
        # Llama-style tables: theta_j = s / 10000**(2j/D) for j < D/2, fp32 [S, D/2].
        half = p["d"] // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) * 2 / p["d"]))
        angles = torch.arange(p["s"], dtype=torch.float32)[:, None] * inv_freq[None, :]
        return [randn(p["b"], p["hq"], p["s"], p["d"]), randn(p["b"], p["hkv"], p["s"], p["d"]),
                torch.cos(angles).contiguous(), torch.sin(angles).contiguous()]
    if task_name == "log_softmax_gather":
        return [randn(p["b"], p["v"]),
                torch.randint(0, p["v"], (p["b"],), dtype=torch.int64)]
    if task_name == "fused_adamw":
        return [randn(p["n"]), randn(p["n"]), randn(p["n"]),
                torch.rand(p["n"], dtype=torch.float32)]
    if task_name == "layernorm_residual_gelu":
        return [randn(p["b"], p["h"]), randn(p["b"], p["h"]),
                randn(p["h"]), randn(p["h"])]
    if task_name == "grpo_group_norm":
        values = randn(p["b"], p["g"])
        values[0] = 1
        return [values]
    if task_name == "cispo_clip_weight":
        log_ratio = torch.empty((p["b"], p["s"]), dtype=torch.float32).uniform_(-8, 8)
        return [log_ratio, randn(p["b"]), torch.rand(p["b"], p["s"]) > 0.2]
    if task_name == "packed_varlen_reduction":
        lengths = torch.randint(0, p["l"] + 1, (p["b"],), dtype=torch.int32)
        lengths[0], lengths[1] = 0, p["l"]
        offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
        return [randn(int(offsets[-1])), offsets]
    if task_name == "flash_attention_fwd":
        shape = (p["b"], p["h"], p["s"], p["d"])
        return [randn(*shape), randn(*shape), randn(*shape)]
    if task_name == "grouped_gemm":
        rows = (8, 32, 64, 128)
        return [[randn(rows[g % 4], p["k"]) for g in range(p["g"])],
                [randn(p["k"], p["n"]) for _ in range(p["g"])]]
    if task_name == "topk_sampling":
        return [randn(p["b"], p["v"]), torch.rand(p["b"], dtype=torch.float32)]
    if task_name == "residual_rmsnorm":
        rows = p["b"] * p["s"]
        return [randn(rows, p["h"]), randn(rows, p["h"]), randn(p["h"])]
    if task_name == "decode_attention":
        return [randn(p["b"], p["hq"], 1, p["d"]),
                randn(p["b"], p["hkv"], p["s"], p["d"]),
                randn(p["b"], p["hkv"], p["s"], p["d"])]
    if task_name == "moe_dense_gemm":
        experts = p["e"] + int(p["shared"])
        x = randn(p["t"], p["h"])
        gate = (torch.randn(experts, p["h"], p["i"]) / p["h"] ** 0.5).to(dtype)
        up = (torch.randn(experts, p["h"], p["i"]) / p["h"] ** 0.5).to(dtype)
        down = (torch.randn(experts, p["i"], p["h"]) / p["i"] ** 0.5).to(dtype)
        ids = torch.rand(p["t"], p["e"]).argsort(dim=1)[:, :p["top_k"]]
        weights = torch.rand(p["t"], p["top_k"])
        weights = weights / weights.sum(dim=1, keepdim=True)
        alpha = torch.zeros(p["t"], experts, dtype=torch.float32)
        alpha.scatter_(1, ids, weights)
        if p["shared"]:
            alpha[:, -1] = 1
        return [x, gate, up, down, alpha]
    if task_name in ("int8_gemm", "fp8_gemm"):
        if task_name == "int8_gemm":
            a = torch.randint(-16, 16, (p["m"], p["k"]), dtype=torch.int8)
            b = torch.randint(-16, 16, (p["k"], p["n"]), dtype=torch.int8)
        else:
            a = torch.randn(p["m"], p["k"]).to(dtype)
            b = torch.randn(p["k"], p["n"]).to(dtype)
        return [a, b, torch.ones(p["m"], dtype=torch.float32),
                torch.ones(p["n"], dtype=torch.float32)]
    raise ValueError(task_name)
