# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; generated with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md; no eagle_harness code copied.
import torch
import triton
import triton.language as tl


@triton.jit
def eagle_ancestor_table(Parents, Paths, N: tl.constexpr, BLOCK: tl.constexpr):
    batch, node = tl.program_id(0), tl.program_id(1)
    current = node
    for depth in range(N):
        tl.store(Paths + (batch*N+node)*N+depth, current)
        current = tl.load(Parents+batch*N+tl.maximum(current,0), current>=0, other=-1)


def paths_for(parents):
    b,n=parents.shape
    paths=torch.empty((b,n,n),device=parents.device,dtype=torch.int32)
    eagle_ancestor_table[(b,n)](parents,paths,n,triton.next_power_of_2(n))
    return paths

@triton.jit
def eagle_vocab_score(X, W, Scores, H: tl.constexpr, V: tl.constexpr, D: tl.constexpr):
    # Vocabulary on grid axis 0 (limit 2**31 - 1); axis 1 is capped at 65535.
    token, batch = tl.program_id(0), tl.program_id(1)
    h=tl.arange(0,D)
    x=tl.load(X+batch*H+h,h<H,other=0).to(tl.float32)
    w=tl.load(W+token*H+h,h<H,other=0).to(tl.float32)
    tl.store(Scores+batch*V+token,tl.sum(x*w,0))


@triton.jit
def eagle_shard_select(Scores, Ids, Values, V: tl.constexpr, K: tl.constexpr, SHARDS: tl.constexpr, BLOCK: tl.constexpr):
    batch,shard=tl.program_id(0),tl.program_id(1)
    ix=shard*BLOCK+tl.arange(0,BLOCK)
    values=tl.load(Scores+batch*V+ix,ix<V,other=-float('inf'))
    for rank in range(K):
        best=tl.max(values,0)
        idx=tl.min(tl.where((values==best)&(ix<V),ix,2147483647),0)
        tl.store(Ids+(batch*SHARDS+shard)*K+rank,idx)
        tl.store(Values+(batch*SHARDS+shard)*K+rank,best)
        values=tl.where(ix==idx,-float('inf'),values)


@triton.jit
def eagle_candidate_merge(Ids, Values, Out, L: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    b=tl.program_id(0)
    ix=tl.arange(0,BLOCK)
    ids=tl.load(Ids+b*L+ix,ix<L,other=2147483647)
    values=tl.load(Values+b*L+ix,ix<L,other=-float('inf'))
    for rank in range(K):
        best=tl.max(values,0)
        idx=tl.min(tl.where(values==best,ids,2147483647),0)
        tl.store(Out+b*K+rank,idx.to(tl.int64))
        values=tl.where(ids==idx,-float('inf'),values)


@triton.jit
def eagle_candidate_rescore(X,W,Ids,Out,H:tl.constexpr,K:tl.constexpr,D:tl.constexpr):
    batch,rank=tl.program_id(0),tl.program_id(1)
    token=tl.load(Ids+batch*K+rank)
    h=tl.arange(0,D)
    x=tl.load(X+batch*H+h,h<H,other=0).to(tl.float32)
    w=tl.load(W+token*H+h,h<H,other=0).to(tl.float32)
    tl.store(Out+batch*K+rank,tl.sum(x*w,0))


def forward(hidden,weight,top_k):
    b,h=hidden.shape;v=weight.shape[0];shards=triton.cdiv(v,1024)
    scores=torch.empty((b,v),device=hidden.device,dtype=torch.float32)
    ids=torch.empty((b,shards*top_k),device=hidden.device,dtype=torch.int32)
    values=torch.empty_like(ids,dtype=torch.float32)
    output=torch.empty((b,top_k),device=hidden.device,dtype=torch.int64)
    exact=torch.empty_like(output,dtype=torch.float32)
    eagle_vocab_score[(v,b)](hidden,weight,scores,h,v,triton.next_power_of_2(h),enable_fp_fusion=False)
    eagle_shard_select[(b,shards)](scores,ids,values,v,top_k,shards,1024)
    eagle_candidate_merge[(b,)](ids,values,output,shards*top_k,top_k,triton.next_power_of_2(shards*top_k))
    eagle_candidate_rescore[(b,top_k)](hidden,weight,output,exact,h,top_k,triton.next_power_of_2(h),enable_fp_fusion=False)
    return output,exact
