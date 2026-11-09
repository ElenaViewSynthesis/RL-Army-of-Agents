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
def eagle_chain_firstchild(Parents,First,N:tl.constexpr,BLOCK:tl.constexpr):
    batch,node=tl.program_id(0),tl.program_id(1)
    ix=tl.arange(0,BLOCK)
    parent=tl.load(Parents+batch*N+ix,ix<N,other=-2)
    child=tl.min(tl.where((parent==node)&(ix<N),ix,N),0)
    tl.store(First+batch*N+node,tl.where(child<N,child,-1))


@triton.jit
def eagle_chain_owner(First,Owner,N:tl.constexpr):
    batch,node=tl.program_id(0),tl.program_id(1)
    current=node
    for _ in range(N):
        child=tl.load(First+batch*N+current)
        current=tl.where(child>=0,child,current)
    tl.store(Owner+batch*N+node,current)


@triton.jit
def eagle_gdn_walk(Q,K,V,G,Initial,Parents,States,Out,N:tl.constexpr,H:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,TILE:tl.constexpr):
    batch,head,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    keydim=tl.arange(0,DK)
    valuedim=tile*TILE+tl.arange(0,TILE)
    for node in range(N):
        parent=tl.load(Parents+batch*N+node)
        initial=tl.load(Initial+((batch*H+head)*DK+keydim[:,None])*DV+valuedim[None,:],valuedim[None,:]<DV,other=0)
        saved=tl.load(States+(((batch*N+tl.maximum(parent,0))*H+head)*DK+keydim[:,None])*DV+valuedim[None,:],(parent>=0)&(valuedim[None,:]<DV),other=0)
        state=tl.where(parent<0,initial,saved)
        offset=(batch*N+node)*H+head
        alpha=tl.load(G+offset*2);beta=tl.load(G+offset*2+1)
        keys=tl.load(K+offset*DK+keydim)
        values=tl.load(V+offset*DV+valuedim,valuedim<DV,other=0)
        queries=tl.load(Q+offset*DK+keydim)
        state=state*alpha
        residual=values-tl.sum(keys[:,None]*state,axis=0)
        state=state+keys[:,None]*(beta*residual)[None,:]
        tl.store(States+(offset*DK+keydim[:,None])*DV+valuedim[None,:],state,valuedim[None,:]<DV)
        tl.store(Out+offset*DV+valuedim,tl.sum(queries[:,None]*state,axis=0),valuedim<DV)


@triton.jit
def eagle_gdn_chain_replay(Q,K,V,G,Initial,Paths,First,Owner,States,Out,N:tl.constexpr,H:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,TILE:tl.constexpr):
    bh,leaf,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    batch,head=bh//H,bh%H
    first=tl.load(First+batch*N+leaf)
    if first<0:
        keydim=tl.arange(0,DK)
        valuedim=tile*TILE+tl.arange(0,TILE)
        state=tl.load(Initial+((batch*H+head)*DK+keydim[:,None])*DV+valuedim[None,:],valuedim[None,:]<DV,other=0)
        for depth in range(N-1,-1,-1):
            node=tl.load(Paths+(batch*N+leaf)*N+depth)
            if node>=0:
                offset=(batch*N+node)*H+head
                alpha=tl.load(G+offset*2);beta=tl.load(G+offset*2+1)
                keys=tl.load(K+offset*DK+keydim)
                values=tl.load(V+offset*DV+valuedim,valuedim<DV,other=0)
                queries=tl.load(Q+offset*DK+keydim)
                state=state*alpha
                residual=values-tl.sum(keys[:,None]*state,axis=0)
                state=state+keys[:,None]*(beta*residual)[None,:]
                owner=tl.load(Owner+batch*N+node)
                if owner==leaf:
                    tl.store(States+(offset*DK+keydim[:,None])*DV+valuedim[None,:],state,valuedim[None,:]<DV)
                    tl.store(Out+offset*DV+valuedim,tl.sum(queries[:,None]*state,axis=0),valuedim<DV)


def forward(q,k,v,gates,initial,parents,chain=False):
    b,n,h,dk=q.shape;dv=v.shape[-1]
    output=torch.empty_like(v)
    states=torch.empty((b,n,h,dk,dv),device=q.device,dtype=torch.float32)
    paths=paths_for(parents)
    if chain:
        first=torch.empty_like(parents);owner=torch.empty_like(parents)
        eagle_chain_firstchild[(b,n)](parents,first,n,triton.next_power_of_2(n))
        eagle_chain_owner[(b,n)](first,owner,n)
        eagle_gdn_chain_replay[(b*h,n,triton.cdiv(dv,16))](q,k,v,gates,initial,paths,first,owner,states,output,n,h,dk,dv,16,enable_fp_fusion=False)
    else:
        eagle_gdn_walk[(b,h,triton.cdiv(dv,16))](q,k,v,gates,initial,parents,states,output,n,h,dk,dv,16,enable_fp_fusion=False)
    return output,states,parents,paths
