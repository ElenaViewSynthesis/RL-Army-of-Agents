"""Independent vectorized per-slot fp32 references; no token-dependent host sync.

Loops are over static slot/slice dimensions only. Never gather T copies of the
adapter matrices: that would exceed H200 memory at the acceptance shape.
"""


def segment_metadata(idx, slots):
    import torch
    t = idx.numel()
    positions = torch.arange(t, device=idx.device)
    keys = torch.where(idx >= 0, idx.long(), slots) * t + positions
    order = torch.argsort(keys, stable=True).to(torch.int32)
    counts = (idx[:, None] == torch.arange(slots, device=idx.device)).sum(0).to(torch.int32)
    slot_order = torch.argsort(torch.where(counts > 0, torch.arange(slots, device=idx.device), slots), stable=True)
    lengths = counts[slot_order]
    ids = torch.where(lengths > 0, slot_order, -1).to(torch.int32)
    starts = torch.where(lengths > 0, lengths.cumsum(0) - lengths, 0).to(torch.int32)
    return order, ids, starts, lengths, (counts > 0).sum().reshape(1).to(torch.int32)


def shrink(x, a, idx, ranks, scales, *metadata, padding=False):
    import torch
    l, s, r, h = a.shape
    output = torch.zeros((s, x.shape[0], r), dtype=torch.float32, device=x.device)
    columns = torch.arange(r, device=x.device)
    for i in range(l):
        mask = columns < ranks[i]
        weights = a[i].float() if padding else torch.where(mask[None, :, None], a[i].float(), 0)
        values = torch.matmul(x.float()[None], weights.transpose(1, 2)) * scales[i]
        values = torch.where(mask[None, None, :], values, 0)
        output = output + torch.where((idx == i)[None, :, None], values, 0)
    return output


def expand(buf, b, y, idx, ranks, offsets, *metadata, padding=False):
    import torch
    l, s, n, r = b.shape
    result = y.clone()
    columns = torch.arange(r, device=y.device)
    # offsets are device inputs; gather destination coordinates without .item().
    for slot in range(l):
        valid = columns < ranks[slot]
        weights = b[slot].float() if padding else torch.where(valid[None, None, :], b[slot].float(), 0)
        safe_buf = buf if padding else torch.where(valid[None, None, :], buf, 0)
        delta = torch.matmul(safe_buf, weights.transpose(1, 2))
        for sl in range(s):
            dest = offsets[sl].long() + torch.arange(n, device=y.device)
            old = result[:, dest]
            value = (old.float() + delta[sl]).to(y.dtype)
            result[:, dest] = torch.where((idx == slot)[:, None], value, old)
    return result


def forward(name, *args, padding=False, half=False):
    if name == "lora_segment_metadata":
        idx, slots_tensor = args
        return segment_metadata(idx, slots_tensor.shape[0])
    if "fused" in name:
        x, a, b, y, idx, ranks, scales, offsets, *meta = args
        buf = shrink(x, a, idx, ranks, scales, *meta, padding=padding)
        if half:
            buf = buf.half().float()
        return expand(buf, b, y, idx, ranks, offsets, *meta, padding=padding)
    if "shrink" in name:
        buf = shrink(*args, padding=padding)
        return buf.half().float() if half else buf
    buf, *rest = args
    return expand(buf.half().float() if half else buf, *rest, padding=padding)


def make_inputs(name, item):
    import torch
    from utils.lora_contracts import slot_ranks, token_slots
    p = item.params
    t, h, n, l, s = (p[k] for k in ("tokens", "hidden", "out", "slots", "slices"))
    r = (p["rank"] + 15) // 16 * 16
    dtype = getattr(torch, item.dtype)
    idx = torch.tensor(token_slots(p), dtype=torch.int32)
    ranks = torch.tensor(slot_ranks(p), dtype=torch.int32)
    if name == "lora_segment_metadata":
        return [idx, ranks]
    scales = torch.linspace(0.5, 1.5, l, dtype=torch.float32)
    offsets = torch.arange(s, dtype=torch.int32) * (n + 16) + 8
    # Adapter dtype conversion happens once here. Padding remains NaN after cast.
    a = (torch.randn(l, s, r, h, dtype=dtype) / h**0.5).to(dtype)
    b = (torch.randn(l, s, n, r, dtype=dtype) / r**0.5).to(dtype)
    rank_mask = torch.arange(r)[None, :] >= ranks[:, None]
    a.masked_fill_(rank_mask[:, None, :, None], float("nan"))
    b.masked_fill_(rank_mask[:, None, None, :], float("nan"))
    x = torch.randn(t, h, dtype=dtype) * 4
    x[idx < 0] = 1024
    y = torch.randn(t, s*(n+16), dtype=dtype)
    y[idx < 0] = 4096
    buf = torch.randn(s, t, r, dtype=torch.float32) * 16
    # Poison intermediate padding too; legitimate paths must mask every rank read.
    token_rank = ranks[idx.clamp_min(0).long()]
    buf.masked_fill_(torch.arange(r)[None, None, :] >= token_rank[None, :, None], float("nan"))
    buf[:, idx < 0] = 1024
    meta = list(segment_metadata(idx, l)) if name.endswith("sgmv") else []
    if "fused" in name:
        return [x, a, b, y, idx, ranks, scales, offsets, *meta]
    if "shrink" in name:
        return [x, a, idx, ranks, scales, *meta]
    return [buf, b, y, idx, ranks, offsets, *meta]
