"""Torch-free Stage 6 contracts, stable grouping and logical work accounting."""

LORA_TASKS = ("lora_shrink_bgmv", "lora_expand_bgmv", "lora_segment_metadata",
              "lora_shrink_sgmv", "lora_expand_sgmv", "lora_fused_sgmv")
RANKS = (8, 16, 32, 64, 128, 256)


def slot_ranks(p):
    return tuple(min(p["rank"], RANKS[i % len(RANKS)]) if p["mixed"] else p["rank"]
                 for i in range(p["slots"]))


def token_slots(p):
    t, l, pattern = p["tokens"], p["slots"], p["pattern"]
    if pattern == "none":
        return [-1] * t
    if pattern == "single":
        return [0] * t
    if pattern == "distinct":
        return [i if i < l else -1 for i in range(t)]
    return [-1 if pattern == "mixed" and i % 4 == 0 else i % l for i in range(t)]


def stage6_tasks(Task, case):
    result = {}
    for name in LORA_TASKS:
        decode = name.endswith("bgmv")
        cases = []
        # Deliberately paired axes, not a Cartesian product. Smallest case first.
        rows = [(1, 2048, 8, 1, 1, "single", False),
                (3, 2048, 8, 4, 1, "mixed", False),
                (1, 4096, 16, 1, 2, "single", False),
                (64, 5120, 32, 16, 3, "round_robin", True),
                (3, 8192, 64, 4, 2, "distinct", False),
                (64, 2048, 128, 16, 1, "none", True),
                (256 if decode else 512, 4096, 256, 16, 3, "mixed", True),
                (64 if decode else 512, 14336, 256, 16, 3, "mixed", True)]
        if not decode:
            rows += [(512, 2048, 64, 4, 1, "single", False)]
            # Acceptance last, max rank 256 with mixed slot ranks.
            rows += [(4096, 14336, 256, 16, 3, "mixed", True)]
        for t, h, r, l, s, pattern, mixed in rows:
            if name.endswith("sgmv"):
                t = max(t, 512)
            for dtype in ("float16", "bfloat16"):
                cases.append(case(dtype, tokens=t, hidden=h, out=h, rank=r,
                                  slots=l, slices=s, pattern=pattern, mixed=mixed))
        result[name] = Task(name, 6,
            "Stable ascending-slot segments, stable token order; skipped tokens last; fixed-size int32 outputs"
            if name == "lora_segment_metadata" else
            "FP32 accumulated multi-LoRA; NaN rank padding masked; skipped rows preserved; disjoint output slices",
            tuple(cases), ("punica", "slora", "vllm"))
    return result


def workload(name, item):
    p = item.params
    ids, ranks = token_slots(p), slot_ranks(p)
    active = sum(i >= 0 for i in ids)
    rank_sum = sum(ranks[i] for i in ids if i >= 0)
    touched = sum(ranks[i] for i in set(ids) if i >= 0)
    s, t, h, n = (p[k] for k in ("slices", "tokens", "hidden", "out"))
    shrink = 2 * s * h * rank_sum
    expand = 2 * s * n * rank_sum
    shrink_bytes = active*h*2 + touched*s*h*2 + s*rank_sum*4
    expand_bytes = s*rank_sum*4 + touched*s*n*2 + active*s*n*4
    if name == "lora_segment_metadata":
        flops, io = 0, 4*(2*t+3*p["slots"]+1)
    elif "fused" in name:
        flops, io = shrink+expand, shrink_bytes+expand_bytes
    elif "shrink" in name:
        flops, io = shrink, shrink_bytes
    else:
        flops, io = expand, expand_bytes
    return {"flops": flops, "logical_io_bytes": io,
            "flop_convention": "2 per active multiply-add; rank padding and skipped tokens excluded",
            "bandwidth_convention": "logical active x, distinct touched weight rows and active buffers; not measured HBM traffic"}
