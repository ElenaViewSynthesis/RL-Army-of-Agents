"""Slow, independent per-token oracle for synthetic tests only; never benchmarked."""


def oracle(name, *args):
    import torch
    if name == "lora_segment_metadata":
        idx, slots = args
        values = idx.tolist()
        order = sorted(range(len(values)), key=lambda t: (values[t] if values[t] >= 0 else len(slots), t))
        active_ids = sorted(set(i for i in values if i >= 0))
        lengths = [values.count(i) for i in active_ids]
        starts = [sum(lengths[:i]) for i in range(len(lengths))]
        padding = len(slots)-len(active_ids)
        return tuple(torch.tensor(x, dtype=torch.int32, device=idx.device) for x in
                     (order, active_ids+[-1]*padding, starts+[0]*padding,
                      lengths+[0]*padding, [len(active_ids)]))
    fused = "fused" in name
    if fused:
        x, a, b, y, idx, ranks, scales, offsets, *meta = args
    elif "shrink" in name:
        x, a, idx, ranks, scales, *meta = args
    else:
        buf, b, y, idx, ranks, offsets, *meta = args
    if fused or "shrink" in name:
        l, s, r, h = a.shape
        buf = torch.zeros((s, len(idx), r), device=x.device)
        for t, slot in enumerate(idx.tolist()):
            if slot < 0:
                continue
            rank = int(ranks[slot])
            for sl in range(s):
                for k in range(rank):
                    buf[sl, t, k] = (x[t].float()*a[slot, sl, k].float()).sum()*scales[slot]
        if not fused:
            return buf
    out = y.clone()
    for t, slot in enumerate(idx.tolist()):
        if slot < 0:
            continue
        rank = int(ranks[slot])
        for sl in range(b.shape[1]):
            off = int(offsets[sl])
            for n in range(b.shape[2]):
                delta = (buf[sl, t, :rank]*b[slot, sl, n, :rank].float()).sum()
                out[t, off+n] = (y[t, off+n].float()+delta).to(y.dtype)
    return out
