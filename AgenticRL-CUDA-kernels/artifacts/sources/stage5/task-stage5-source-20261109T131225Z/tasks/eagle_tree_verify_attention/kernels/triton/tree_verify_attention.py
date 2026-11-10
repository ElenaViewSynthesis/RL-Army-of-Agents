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
def eagle_attention_splits(Q,K,V,TK,TV,Paths,Part,Maximum,Denom,N:tl.constexpr,S:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,WINDOW:tl.constexpr,SPLITS:tl.constexpr,DEPTH:tl.constexpr,BLOCK:tl.constexpr):
    bh,node,split=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    batch,head=bh//HQ,bh%HQ
    kvhead=head//(HQ//HK)
    index=split*BLOCK+tl.arange(0,BLOCK)
    committed=index<WINDOW
    tid=index-WINDOW
    columns=tl.arange(0,D)
    q=tl.load(Q+((batch*HQ+head)*N+node)*D+columns).to(tl.float32)
    ki=((batch*HK+kvhead)*S+S-WINDOW+index[:,None])*D+columns[None,:]
    ti=((batch*HK+kvhead)*N+tl.maximum(tid[:,None],0))*D+columns[None,:]
    key=tl.load(K+ki,committed[:,None],other=0).to(tl.float32)+tl.load(TK+ti,((tid>=0)&(tid<N))[:,None],other=0).to(tl.float32)
    value=tl.load(V+ki,committed[:,None],other=0).to(tl.float32)+tl.load(TV+ti,((tid>=0)&(tid<N))[:,None],other=0).to(tl.float32)
    depth=tl.arange(0,DEPTH)
    ancestors=tl.load(Paths+(batch*N+node)*N+depth,depth<N,other=-1)
    visible=tl.sum(((tid[:,None]==ancestors[None,:])&(ancestors[None,:]>=0)).to(tl.int32),axis=1)>0
    logits=tl.sum(key*q[None,:],axis=1)*(D**-.5)
    logits=tl.where(committed|((tid<N)&visible),logits,-float('inf'))
    maximum=tl.max(logits,0)
    safe=tl.where(maximum==-float('inf'),0.,maximum)
    probs=tl.exp(logits-safe)
    denom=tl.sum(probs,0)
    weighted=tl.sum(probs[:,None]*value,axis=0)
    offset=(bh*N+node)*SPLITS+split
    tl.store(Part+offset*D+columns,weighted)
    tl.store(Maximum+offset,maximum)
    tl.store(Denom+offset,denom)


@triton.jit
def eagle_attention_merge(Part,Maximum,Denom,Out,N:tl.constexpr,D:tl.constexpr,SPLITS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    splits=tl.arange(0,BLOCK)
    columns=tl.arange(0,D)
    maximum=tl.load(Maximum+row*SPLITS+splits,splits<SPLITS,other=-float('inf'))
    globalmax=tl.max(maximum,0)
    scale=tl.exp(maximum-globalmax)
    denom=tl.load(Denom+row*SPLITS+splits,splits<SPLITS,other=0)
    parts=tl.load(Part+(row*SPLITS+splits[:,None])*D+columns[None,:],(splits<SPLITS)[:,None],other=0)
    output=tl.sum(parts*scale[:,None],axis=0)/tl.sum(denom*scale,0)
    tl.store(Out+row*D+columns,output)


def forward(q,k,v,tk,tv,parents,window):
    b,hq,n,d=q.shape;hk,s=k.shape[1:3]
    splits=triton.cdiv(window+n,128)
    paths=paths_for(parents)
    part=torch.empty((b*hq*n,splits,d),device=q.device,dtype=torch.float32)
    maximum=torch.empty((b*hq*n,splits),device=q.device,dtype=torch.float32)
    denom=torch.empty_like(maximum)
    output=torch.empty_like(q)
    eagle_attention_splits[(b*hq,n,splits)](q,k,v,tk,tv,paths,part,maximum,denom,n,s,hq,hk,d,window,splits,triton.next_power_of_2(n),128,enable_fp_fusion=False)
    eagle_attention_merge[(b*hq*n,)](part,maximum,denom,output,n,d,splits,triton.next_power_of_2(splits),enable_fp_fusion=False)
    return output,paths
