# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# Tiled tensor-core rewrite of the attention kernels, 2026-11-14 (Claude).
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
"""Tree attention over committed KV plus self-inclusive ancestor KV.

One program handles a tile of BLOCK_M query rows that share one KV head: the
RATIO query heads of a GQA group times the N tree nodes are contiguous in Q, so
each committed key and value tile is loaded once per tile of rows instead of
once per row. Scores and P @ V run as tl.dot on bf16 operands with fp32
accumulation (tensor cores on sm_80+); the softmax is fp32 and online. The
committed window is split across programs (flash-decoding); the tree keys get
their own split with the ancestor mask; a merge kernel combines the splits'
maxima, normalisers and weighted sums. Tile sizes are fixed per shape, not
autotuned.
"""
import os

import torch
import triton
import triton.language as tl

BLOCK_N = 64            # committed keys per inner step
MAX_BLOCK_M = 128       # query rows per program
MIN_PROGRAMS = 1024     # committed splits shrink (down to MIN_SPLIT) until the grid has this many
MAX_SPLIT, MIN_SPLIT = 4096, 512

# Triton's CPU interpreter computes tl.dot on bf16 operands from their raw bits
# (checked on triton 3.8.0), so under TRITON_INTERPRET=1 the operands are
# upcast to fp32 first. bf16 -> fp32 is exact, so the interpreter computes the
# same products; the GPU path keeps bf16 operands for the tensor cores.
UPCAST = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def eagle_mma(a, b, UPCAST: tl.constexpr):
    if UPCAST:
        out = tl.dot(a.to(tl.float32), b.to(tl.float32))
    else:
        out = tl.dot(a, b)
    return out


@triton.jit
def eagle_ancestor_table(Parents, Paths, N: tl.constexpr, BLOCK: tl.constexpr):
    batch, node = tl.program_id(0), tl.program_id(1)
    current = node
    for depth in range(N):
        tl.store(Paths + (batch*N+node)*N+depth, current)
        current = tl.load(Parents+batch*N+tl.maximum(current,0), current>=0, other=-1)


@triton.jit
def eagle_visible_table(Paths, Visible, N: tl.constexpr, BLOCK: tl.constexpr):
    # Visible[b, node, j] = 1 when tree key j is node itself or one of its ancestors.
    batch, node = tl.program_id(0), tl.program_id(1)
    depth = tl.arange(0, BLOCK)
    ancestors = tl.load(Paths + (batch*N+node)*N + depth, depth < N, other=-1)
    keys = tl.arange(0, BLOCK)
    hit = (ancestors[None, :] == keys[:, None]) & (ancestors[None, :] >= 0)
    visible = tl.sum(hit.to(tl.int32), axis=1) > 0
    tl.store(Visible + (batch*N+node)*N + keys, visible.to(tl.int8), keys < N)


def paths_for(parents):
    b,n=parents.shape
    paths=torch.empty((b,n,n),device=parents.device,dtype=torch.int32)
    eagle_ancestor_table[(b,n)](parents,paths,n,triton.next_power_of_2(n))
    return paths


@triton.jit
def eagle_attention_committed(Q, K, V, Part, Maximum, Denom,
                              N: tl.constexpr, S: tl.constexpr, W: tl.constexpr, HQ: tl.constexpr,
                              HK: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
                              SPLIT_LEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                              UPCAST: tl.constexpr):
    # int64 offsets: (b*hk*s)*d and (b*hq*n*splits)*d grow past 2**31 at 32k contexts.
    group_id = tl.program_id(0).to(tl.int64)
    tile, split = tl.program_id(1), tl.program_id(2)
    RATIO: tl.constexpr = HQ // HK
    batch, kvhead = group_id // HK, group_id % HK
    rows = tile*BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows < RATIO*N
    columns = tl.arange(0, D)
    first = (batch*HQ + kvhead*RATIO)*N  # flat (head, node) row of the group's first query
    q = tl.load(Q + (first + rows[:, None])*D + columns[None, :], valid[:, None], other=0.)
    base = (group_id*S + S - W)*D      # the last W committed keys of this KV head
    start = split*SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, W)
    maximum = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    denom = tl.zeros((BLOCK_M,), tl.float32)
    weighted = tl.zeros((BLOCK_M, D), tl.float32)
    keys = tl.arange(0, BLOCK_N)
    for offset in range(start, end, BLOCK_N):
        position = offset + keys
        inside = position < end
        index = base + position[:, None]*D + columns[None, :]
        key = tl.load(K + index, inside[:, None], other=0.)
        value = tl.load(V + index, inside[:, None], other=0.)
        logits = eagle_mma(q, tl.trans(key), UPCAST) * (D**-.5)
        logits = tl.where(inside[None, :], logits, -float('inf'))
        new = tl.maximum(maximum, tl.max(logits, 1))
        alpha = tl.exp(maximum - new)
        probs = tl.exp(logits - new[:, None])
        denom = denom*alpha + tl.sum(probs, 1)
        weighted = weighted*alpha[:, None] + eagle_mma(probs.to(value.dtype), value, UPCAST)
        maximum = new
    offset = (first + rows)*SPLITS + split
    tl.store(Part + offset[:, None]*D + columns[None, :], weighted, valid[:, None])
    tl.store(Maximum + offset, maximum, valid)
    tl.store(Denom + offset, denom, valid)


@triton.jit
def eagle_attention_tree(Q, TK, TV, Visible, Part, Maximum, Denom,
                         N: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                         SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr,
                         UPCAST: tl.constexpr):
    # The last split: this group's N tree keys, each row masked to its own ancestors.
    group_id = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    RATIO: tl.constexpr = HQ // HK
    batch, kvhead = group_id // HK, group_id % HK
    rows = tile*BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows < RATIO*N
    node = rows % N
    columns = tl.arange(0, D)
    first = (batch*HQ + kvhead*RATIO)*N
    q = tl.load(Q + (first + rows[:, None])*D + columns[None, :], valid[:, None], other=0.)
    keys = tl.arange(0, BLOCK_T)
    real = keys < N
    index = (group_id*N + keys[:, None])*D + columns[None, :]
    key = tl.load(TK + index, real[:, None], other=0.)
    value = tl.load(TV + index, real[:, None], other=0.)
    logits = eagle_mma(q, tl.trans(key), UPCAST) * (D**-.5)
    seen = tl.load(Visible + (batch*N + node[:, None])*N + keys[None, :],
                   valid[:, None] & real[None, :], other=0) != 0
    logits = tl.where(seen, logits, -float('inf'))
    maximum = tl.max(logits, 1)
    safe = tl.where(maximum == -float('inf'), 0., maximum)
    probs = tl.exp(logits - safe[:, None])
    denom = tl.sum(probs, 1)
    weighted = eagle_mma(probs.to(value.dtype), value, UPCAST)
    offset = (first + rows)*SPLITS + SPLITS - 1
    tl.store(Part + offset[:, None]*D + columns[None, :], weighted, valid[:, None])
    tl.store(Maximum + offset, maximum, valid)
    tl.store(Denom + offset, denom, valid)


@triton.jit
def eagle_attention_merge(Part,Maximum,Denom,Out,N:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    splits=tl.arange(0,BLOCK)
    columns=tl.arange(0,D)
    maximum=tl.load(Maximum+row*SPLITS+splits,splits<SPLITS,other=-float('inf'))
    globalmax=tl.max(maximum,0)
    scale=tl.exp(maximum-globalmax)
    denom=tl.load(Denom+row*SPLITS+splits,splits<SPLITS,other=0)
    parts=tl.load(Part+(row*SPLITS+splits[:,None])*D+columns[None,:],(splits<SPLITS)[:,None],other=0)
    output=tl.sum(parts*scale[:,None],axis=0)/tl.sum(denom*scale,0)
    tl.store(Out+row*D+columns,output)


def plan(b, hq, hk, n, s, window):
    """Fixed launch shape for one case: (BLOCK_M, row tiles, committed window, split length, committed splits)."""
    rows = (hq // hk) * n
    block_m = min(MAX_BLOCK_M, max(16, triton.next_power_of_2(rows)))
    tiles = triton.cdiv(rows, block_m)
    width = min(window, s)
    split = MAX_SPLIT
    while split > MIN_SPLIT and b * hk * tiles * triton.cdiv(width, split) < MIN_PROGRAMS:
        split //= 2
    return block_m, tiles, width, split, triton.cdiv(width, split)


def forward(q,k,v,tk,tv,parents,window):
    b,hq,n,d=q.shape;hk,s=k.shape[1:3]
    block_m, tiles, width, split, committed = plan(b, hq, hk, n, s, window)
    splits = committed + 1
    warps = 8 if block_m >= 128 else 4
    paths=paths_for(parents)
    visible = torch.empty((b,n,n),device=q.device,dtype=torch.int8)
    eagle_visible_table[(b,n)](paths,visible,n,triton.next_power_of_2(n))
    part=torch.empty((b*hq*n,splits,d),device=q.device,dtype=torch.float32)
    maximum=torch.empty((b*hq*n,splits),device=q.device,dtype=torch.float32)
    denom=torch.empty_like(maximum)
    output=torch.empty_like(q)
    eagle_attention_committed[(b*hk, tiles, committed)](
        q, k, v, part, maximum, denom, n, s, width, hq, hk, d, splits, split, block_m, BLOCK_N,
        UPCAST, num_warps=warps, num_stages=2)
    eagle_attention_tree[(b*hk, tiles)](
        q, tk, tv, visible, part, maximum, denom, n, hq, hk, d, splits, block_m,
        max(16, triton.next_power_of_2(n)), UPCAST, num_warps=warps)
    eagle_attention_merge[(b*hq*n,)](part,maximum,denom,output,n,d,splits,triton.next_power_of_2(splits),enable_fp_fusion=False)
    return output,paths
