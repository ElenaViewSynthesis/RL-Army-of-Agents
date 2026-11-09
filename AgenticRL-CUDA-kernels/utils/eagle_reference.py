# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; authored with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md.
"""Trusted Torch references for the explicit Stage 5 contracts.

Derived from documented behavior only; no eagle_harness code was inspected.
Eagle 4 Source-Available License; see legal/eagle4/ and EAGLE4_STAGE5.md.
"""


def ancestors(parents):
    """[b, n, n] ancestor paths (node, parent, ..., -1 padded), vectorized over the batch.

    Loops only over depth (n <= 64) with gathers on the device: no host syncs,
    so the eager baseline is a fair GPU reference. Matches the original
    per-element walk exactly (tests/test_eagle_reference_vectorized.py).
    """
    import torch
    b, n = parents.shape
    links = parents.long()
    rows = torch.empty((b, n, n), device=parents.device, dtype=torch.int32)
    current = torch.arange(n, device=parents.device).expand(b, n)
    for depth in range(n):
        rows[:, :, depth] = current
        current = torch.where(current >= 0, links.gather(1, current.clamp(min=0)),
                              torch.full_like(current, -1))
    return rows


def candidate_topk(hidden, weight, top_k):
    import torch
    scores = hidden.float() @ weight.float().T
    ids = scores.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    # Re-score with fp32 products (not bf16 product rounding).
    exact = (hidden.float()[:, None, :] * weight[ids].float()).sum(dim=-1)
    return ids, exact


def tree_step(scores, children, parents):
    import torch
    b, n, k = children.shape
    total = (scores[:, :, None] + children).flatten(1)
    ids = total.argsort(dim=1, descending=True, stable=True)[:, :n]
    parent_ids = (ids // k).to(torch.int32)
    selected = total.gather(1, ids)
    old = ancestors(parents)
    paths = torch.empty((b, n, n+1), dtype=torch.int32, device=parents.device)
    paths[:, :, 0] = torch.arange(n, 2*n, device=parents.device, dtype=torch.int32)
    paths[:, :, 1:] = old.gather(1, parent_ids.long()[:, :, None].expand(-1, -1, n))
    return selected, parent_ids, paths, ids.to(torch.int32)


def attention(q, committed_k, committed_v, tree_k, tree_v, parents, window):
    import torch
    b, hq, n, d = q.shape
    hkv, s = committed_k.shape[1:3]
    paths = ancestors(parents)
    positions = torch.arange(n, device=q.device)
    visible = (paths[:, :, :, None] == positions).any(dim=2)
    head = torch.arange(hq, device=q.device) // (hq // hkv)
    key = torch.cat((committed_k[:, head, -window:], tree_k[:, head]), dim=2).float()
    value = torch.cat((committed_v[:, head, -window:], tree_v[:, head]), dim=2).float()
    logits = (q.float() @ key.transpose(-1, -2)) / d**.5
    mask = torch.cat((torch.ones((b, n, min(window, s)), dtype=torch.bool, device=q.device), visible), dim=-1)
    logits = logits.masked_fill(~mask[:, None], -float('inf'))
    out = logits.softmax(dim=-1) @ value
    return out.to(q.dtype), paths


def tree_conv(x, history, weight, bias, parents, mamba=False):
    """Width-4 tree causal conv1d, vectorized over batch, nodes and channels.

    Same tap order and fp32 accumulation as the original per-node loop; window
    positions < 0 read history[current + 3].
    """
    import torch
    tokens = x.transpose(1, 2) if mamba else x
    b, n, c = tokens.shape
    links = parents.long()
    windows = torch.empty((b, n, 4), dtype=torch.int32, device=x.device)
    total = torch.zeros((b, n, c), dtype=torch.float32, device=x.device)
    current = torch.arange(n, device=x.device).expand(b, n)
    for offset in range(4):
        windows[:, :, offset] = current
        valid = current >= 0
        own = tokens.gather(1, current.clamp(min=0)[:, :, None].expand(-1, -1, c)).float()
        past = history.gather(1, (current + 3).clamp(0, 2)[:, :, None].expand(-1, -1, c)).float()
        source = torch.where(valid[:, :, None], own, past)
        total = total + source * weight[:, 3 - offset]
        current = torch.where(valid, links.gather(1, current.clamp(min=0)), current - 1)
    output = total + bias
    return (output.transpose(1, 2).contiguous() if mamba else output), parents.clone(), windows


def recurrence(q, key, value, gates, initial, parents):
    """Tree gated delta rule; loops over nodes only, parent states gathered on device."""
    import torch
    b, n, heads, dk = q.shape
    dv = value.shape[-1]
    batch = torch.arange(b, device=q.device)
    states = torch.empty((b, n, heads, dk, dv), dtype=torch.result_type(initial, value), device=q.device)
    outputs = []
    for node in range(n):
        parent = parents[:, node].long()
        from_parent = states[batch, parent.clamp(min=0)]
        base = torch.where((parent < 0)[:, None, None, None], initial.to(states.dtype), from_parent)
        decayed = base * gates[:, node, :, 0, None, None]
        residual = value[:, node] - (key[:, node, :, :, None] * decayed).sum(dim=-2)
        delta = gates[:, node, :, 1, None] * residual
        updated = decayed + key[:, node, :, :, None] * delta[:, :, None, :]
        states[:, node] = updated
        outputs.append((q[:, node, :, :, None] * updated).sum(dim=-2))
    return torch.stack(outputs, dim=1), states, parents.clone(), ancestors(parents)


def _walk(choose, slots, b, n, device):
    """Shared greedy tree walk: choose(current) -> (found, child) for all rows at once."""
    import torch
    current = torch.full((b,), -1, dtype=torch.long, device=device)
    active = torch.ones(b, dtype=torch.bool, device=device)
    lengths = torch.zeros(b, dtype=torch.int32, device=device)
    paths = torch.full((b, n), -1, dtype=slots.dtype, device=device)
    compact = torch.full_like(slots, -1)
    batch = torch.arange(b, device=device)
    for step in range(n):
        found, child = choose(current)
        take = active & found
        paths[:, step] = torch.where(take, child.to(paths.dtype), paths[:, step])
        compact[:, step] = torch.where(take, slots[batch, child.clamp(min=0)], compact[:, step])
        current = torch.where(take, child, current)
        lengths += take.to(torch.int32)
        active = take
    return current, lengths, paths, compact


def accept(tokens, parents, targets, slots):
    """Greedy acceptance, vectorized over the batch (first matching child wins)."""
    import torch
    b, n = tokens.shape
    batch = torch.arange(b, device=tokens.device)

    def choose(current):
        wanted = targets[batch, current + 1]
        match = (parents.long() == current[:, None]) & (tokens == wanted[:, None])
        return match.any(dim=1), match.to(torch.int8).argmax(dim=1)

    current, lengths, paths, compact = _walk(choose, slots, b, n, tokens.device)
    paths = paths.to(tokens.dtype)
    bonus = targets[batch, current + 1].to(torch.int32)
    return lengths, paths, bonus, compact, parents.clone()


def accept_sampled(tokens, parents, target, draft, slots, uniforms, bonus_uniforms):
    """Seeded rejection-sampling acceptance; loops over children only.

    Every edge is evaluated in child order for a fixed work count, as before;
    only active rejections commit a residual update.
    """
    import torch
    b, n = tokens.shape
    device = tokens.device
    batch = torch.arange(b, device=device)
    residual = target.float().clone()
    winner = torch.full((b, n + 1), -1, device=device, dtype=torch.int32)
    for child in range(n):
        row = parents[:, child].long() + 1
        token = tokens[:, child].long()
        proposal = draft[batch, row]
        current = residual[batch, row]
        p_token = proposal[batch, token]
        ratio = torch.where(p_token > 0, (current[batch, token] / p_token).clamp(max=1.),
                            torch.zeros_like(p_token))
        rejected = uniforms[:, child] >= ratio
        corrected = (current - proposal).clamp_min(0)
        mass = corrected.sum(dim=-1, dtype=torch.float64).float()
        corrected = corrected / torch.where(mass > 0, mass, torch.ones_like(mass))[:, None]
        open_row = winner[batch, row] < 0
        residual[batch, row] = torch.where((open_row & rejected)[:, None], corrected, current)
        winner[batch, row] = torch.where(open_row & ~rejected, torch.full_like(row, child, dtype=torch.int32),
                                         winner[batch, row])
    cdf = residual.cumsum(dim=-1, dtype=torch.float64)
    bonuses = (cdf <= bonus_uniforms[:, :, None].double()).sum(dim=-1).clamp_max(target.shape[-1] - 1)
    probability = residual.gather(2, bonuses[:, :, None]).squeeze(-1)

    def choose(current):
        child = winner[batch, current + 1].long()
        return child >= 0, child

    current, lengths, paths, compact = _walk(choose, slots, b, n, device)
    paths = paths.to(tokens.dtype)
    bonus = bonuses[batch, current + 1].to(torch.int32)
    bonus_probability = probability[batch, current + 1]
    return lengths, paths, bonus, compact, parents.clone(), bonus_probability


def make_inputs(name, item):
    import torch
    from utils.eagle_contracts import tree_parents
    p = item.params
    b = p['b']
    if name == 'eagle_candidate_topk':
        return [torch.randn(b, p['h'], dtype=torch.bfloat16),
                torch.randn(p['v'], p['h'], dtype=torch.bfloat16)]
    n = p['n']
    parents = torch.tensor([tree_parents(n, p['tree'])]*b, dtype=torch.int32)
    if name == 'eagle_tree_step':
        return [torch.randn(b,n), torch.randn(b,n,p['top_k']), parents]
    if name in ('eagle_tree_row_attention','eagle_tree_verify_attention'):
        hkv,hq,d,s = p['hkv'],p['hkv']*p['ratio'],p['d'],p['s']
        return [torch.randn(b,hq,n,d,dtype=torch.bfloat16),
                torch.randn(b,hkv,s,d,dtype=torch.bfloat16),torch.randn(b,hkv,s,d,dtype=torch.bfloat16),
                torch.randn(b,hkv,n,d,dtype=torch.bfloat16),torch.randn(b,hkv,n,d,dtype=torch.bfloat16),parents]
    if name in ('eagle_gdn_tree_conv','eagle_mamba2_tree_conv'):
        c=p['c']
        x=torch.randn(b,n,c,dtype=torch.bfloat16)
        if name=='eagle_mamba2_tree_conv':
            x=x.transpose(1,2).contiguous()
        return [x,torch.randn(b,3,c,dtype=torch.bfloat16),torch.randn(c,4),torch.randn(c),parents]
    if name=='eagle_gdn_tree_recurrence':
        h,k,v=p['heads'],p['dk'],p['dv']
        gates=torch.rand(b,n,h,2)
        gates[...,0]=.8+.19*gates[...,0]
        gates[...,1]=.1+.4*gates[...,1]
        return [torch.randn(b,n,h,k)/k**.5,torch.randn(b,n,h,k)/k**.5,
                torch.randn(b,n,h,v),gates,torch.randn(b,h,k,v)*.1,parents]
    v=p.get('v',32000)
    slots=torch.arange(n,dtype=torch.int32)[None,:].expand(b,-1).contiguous()+100000
    if name=='eagle_accept_tail':
        tokens=torch.randint(0,v,(b,n),dtype=torch.int32)
        targets=torch.randint(0,v,(b,n+1),dtype=torch.int32)
        # Exercise both accepted chains and a rejection; never depend on this in candidates.
        for node in range(n):
            parent=int(parents[0,node])
            if node%3 != 2: targets[:,parent+1]=tokens[:,node]
        return [tokens,parents,targets,slots]
    target=torch.softmax(torch.randn(b,n+1,v),dim=-1)
    draft=torch.softmax(torch.randn(b,n+1,v),dim=-1)
    tokens=torch.empty(b,n,dtype=torch.int32)
    for node in range(n):
        tokens[:,node]=torch.multinomial(draft[:,int(parents[0,node])+1],1).flatten().to(torch.int32)
    rng=torch.Generator().manual_seed(p['seed'])
    return [tokens,parents,target,draft,slots,torch.rand(b,n,generator=rng),torch.rand(b,n+1,generator=rng)]
