# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; authored with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md.
"""Trusted Torch references for the explicit Stage 5 contracts.

Derived from documented behavior only; no eagle_harness code was inspected.
Eagle 4 Source-Available License; see legal/eagle4/ and EAGLE4_STAGE5.md.
"""


def ancestors(parents):
    import torch
    b, n = parents.shape
    rows = torch.empty((b, n, n), device=parents.device, dtype=torch.int32)
    for batch in range(b):
        for node in range(n):
            current = node
            for depth in range(n):
                rows[batch, node, depth] = current
                current = int(parents[batch, current]) if current >= 0 else -1
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
    import torch
    tokens = x.transpose(1, 2) if mamba else x
    b, n, c = tokens.shape
    windows = torch.empty((b, n, 4), dtype=torch.int32, device=x.device)
    output = torch.empty((b, n, c), dtype=torch.float32, device=x.device)
    for batch in range(b):
        for node in range(n):
            current, total = node, torch.zeros(c, device=x.device)
            for offset in range(4):
                windows[batch, node, offset] = current
                source = tokens[batch, current].float() if current >= 0 else history[batch, current+3].float()
                total = total + source * weight[:, 3-offset]
                current = int(parents[batch, current]) if current >= 0 else current-1
            output[batch, node] = total + bias
    return (output.transpose(1, 2).contiguous() if mamba else output), parents.clone(), windows


def recurrence(q, key, value, gates, initial, parents):
    import torch
    b, n, heads, dk = q.shape
    dv = value.shape[-1]
    states, outputs = [], []
    for node in range(n):
        base = torch.stack([initial[batch] if int(parents[batch, node]) < 0
                            else states[int(parents[batch, node])][batch] for batch in range(b)])
        decayed = base * gates[:, node, :, 0, None, None]
        residual = value[:, node] - (key[:, node, :, :, None] * decayed).sum(dim=-2)
        delta = gates[:, node, :, 1, None] * residual
        updated = decayed + key[:, node, :, :, None] * delta[:, :, None, :]
        states.append(updated)
        outputs.append((q[:, node, :, :, None] * updated).sum(dim=-2))
    return torch.stack(outputs, dim=1), torch.stack(states, dim=1), parents.clone(), ancestors(parents)


def accept(tokens, parents, targets, slots):
    import torch
    b, n = tokens.shape
    lengths = torch.zeros(b, dtype=torch.int32, device=tokens.device)
    paths = torch.full_like(tokens, -1)
    compact = torch.full_like(slots, -1)
    bonus = torch.empty(b, dtype=torch.int32, device=tokens.device)
    for batch in range(b):
        current, length = -1, 0
        while length < n:
            wanted = int(targets[batch, current+1])
            choices = ((parents[batch] == current) & (tokens[batch] == wanted)).nonzero()
            if len(choices) == 0:
                break
            current = int(choices[0, 0])
            paths[batch, length], compact[batch, length] = current, slots[batch, current]
            length += 1
        lengths[batch], bonus[batch] = length, targets[batch, current+1]
    return lengths, paths, bonus, compact, parents.clone()


def accept_sampled(tokens, parents, target, draft, slots, uniforms, bonus_uniforms):
    import torch
    b, n = tokens.shape
    residual = target.float().clone()
    winner = torch.full((b,n+1), -1, device=tokens.device, dtype=torch.int32)
    # Evaluate every edge, even after a parent's first acceptance, for a fixed
    # algorithmic work count. Only active rejections commit a residual update.
    for child in range(n):
        for batch in range(b):
            row = int(parents[batch,child])+1
            token = int(tokens[batch,child])
            proposal = draft[batch,row]
            ratio = min(1., float(residual[batch,row,token] / proposal[token])) if proposal[token] > 0 else 0.
            rejected = float(uniforms[batch,child]) >= ratio
            corrected = (residual[batch,row]-proposal).clamp_min(0)
            mass = corrected.sum(dtype=torch.float64).float()
            corrected = corrected / torch.where(mass>0,mass,torch.ones_like(mass))
            if int(winner[batch,row]) < 0:
                if rejected:
                    residual[batch,row] = corrected
                else:
                    winner[batch,row] = child
    cdf = residual.cumsum(dim=-1, dtype=torch.float64)
    bonuses = (cdf <= bonus_uniforms[:,:,None].double()).sum(dim=-1).clamp_max(target.shape[-1]-1)
    probability = residual.gather(2,bonuses[:,:,None]).squeeze(-1)
    lengths = torch.zeros(b,dtype=torch.int32,device=tokens.device)
    paths, compact = torch.full_like(tokens,-1),torch.full_like(slots,-1)
    bonus = torch.empty(b,dtype=torch.int32,device=tokens.device)
    bonus_probability = torch.empty(b,device=tokens.device)
    for batch in range(b):
        current,length = -1,0
        while length<n:
            child = int(winner[batch,current+1])
            if child<0:
                break
            paths[batch,length],compact[batch,length]=child,slots[batch,child]
            current,length=child,length+1
        lengths[batch]=length
        bonus[batch]=bonuses[batch,current+1]
        bonus_probability[batch]=probability[batch,current+1]
    return lengths,paths,bonus,compact,parents.clone(),bonus_probability


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
