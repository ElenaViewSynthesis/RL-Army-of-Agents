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
def eagle_sample_init(Target,Residual,Winner,Bonuses,N:tl.constexpr,V:tl.constexpr,BLOCK:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    ix=tile*BLOCK+tl.arange(0,BLOCK)
    values=tl.load(Target+row*V+ix,ix<V,other=0)
    tl.store(Residual+row*V+ix,values,ix<V)
    if tile==0:
        tl.store(Winner+row,-1)
        tl.store(Bonuses+row,V-1)


@triton.jit
def eagle_rejection_decision(Tokens,Parents,Draft,Residual,Uniforms,Winner,Flags,N:tl.constexpr,V:tl.constexpr,CHILD:tl.constexpr):
    batch=tl.program_id(0)
    row=tl.load(Parents+batch*N+CHILD)+1
    token=tl.load(Tokens+batch*N+CHILD)
    offset=(batch*(N+1)+row)*V+token
    q=tl.load(Draft+offset);p=tl.load(Residual+offset)
    ratio=tl.minimum(1.,tl.div_rn(p,tl.where(q>0,q,1.)))
    ratio=tl.where(q>0,ratio,0.)
    active=tl.load(Winner+batch*(N+1)+row)<0
    reject=tl.load(Uniforms+batch*N+CHILD)>=ratio
    tl.store(Flags+batch,active&reject)
    if active&(~reject):
        tl.store(Winner+batch*(N+1)+row,CHILD)


@triton.jit
def eagle_residual_chunks(Parents,Draft,Residual,Scratch,Mass,N:tl.constexpr,V:tl.constexpr,CHILD:tl.constexpr,TILES:tl.constexpr,BLOCK:tl.constexpr):
    batch,tile=tl.program_id(0),tl.program_id(1)
    row=tl.load(Parents+batch*N+CHILD)+1
    ix=tile*BLOCK+tl.arange(0,BLOCK)
    offset=(batch*(N+1)+row)*V+ix
    p=tl.load(Residual+offset,ix<V,other=0)
    q=tl.load(Draft+offset,ix<V,other=0)
    corrected=tl.maximum(p-q,0.)
    tl.store(Scratch+batch*V+ix,corrected,ix<V)
    tl.store(Mass+batch*TILES+tile,tl.sum(corrected.to(tl.float64),0))


@triton.jit
def eagle_residual_normalize(Parents,Residual,Scratch,Mass,Flags,N:tl.constexpr,V:tl.constexpr,CHILD:tl.constexpr,TILES:tl.constexpr,MASSBLOCK:tl.constexpr,BLOCK:tl.constexpr):
    batch,tile=tl.program_id(0),tl.program_id(1)
    row=tl.load(Parents+batch*N+CHILD)+1
    mi=tl.arange(0,MASSBLOCK)
    mass=tl.sum(tl.load(Mass+batch*TILES+mi,mi<TILES,other=0),0).to(tl.float32)
    ix=tile*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(Scratch+batch*V+ix,ix<V,other=0)
    value=tl.div_rn(value,tl.where(mass>0,mass,1.))
    flag=tl.load(Flags+batch)
    tl.store(Residual+(batch*(N+1)+row)*V+ix,value,(ix<V)&flag)


@triton.jit
def eagle_probability_masses(Residual,Mass,V:tl.constexpr,TILES:tl.constexpr,BLOCK:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    ix=tile*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(Residual+row*V+ix,ix<V,other=0).to(tl.float64)
    tl.store(Mass+row*TILES+tile,tl.sum(value,0))


@triton.jit
def eagle_bonus_sample(Residual,Mass,Uniforms,Bonuses,V:tl.constexpr,TILES:tl.constexpr,MASSBLOCK:tl.constexpr,BLOCK:tl.constexpr):
    row,tile=tl.program_id(0),tl.program_id(1)
    mi=tl.arange(0,MASSBLOCK)
    prefix=tl.sum(tl.load(Mass+row*TILES+mi,mi<tile,other=0),0)
    ix=tile*BLOCK+tl.arange(0,BLOCK)
    values=tl.load(Residual+row*V+ix,ix<V,other=0).to(tl.float64)
    cdf=tl.cumsum(values,0)+prefix
    uniform=tl.load(Uniforms+row).to(tl.float64)
    selected=tl.min(tl.where((cdf>uniform)&(ix<V),ix,V-1),0)
    tl.atomic_min(Bonuses+row,selected)


@triton.jit
def eagle_sampled_walk(Parents,Slots,Winner,Bonuses,Residual,Lengths,Paths,Bonus,Compact,Probability,N:tl.constexpr,V:tl.constexpr,BLOCK:tl.constexpr):
    batch=tl.program_id(0)
    ix=tl.arange(0,BLOCK)
    tl.store(Paths+batch*N+ix,-1,ix<N)
    tl.store(Compact+batch*N+ix,-1,ix<N)
    current=-1;length=0;active=1
    for _ in range(N):
        child=tl.load(Winner+batch*(N+1)+current+1)
        if (active==1)&(child>=0):
            tl.store(Paths+batch*N+length,child)
            tl.store(Compact+batch*N+length,tl.load(Slots+batch*N+child))
            current=child;length+=1
        else:
            active=0
    row=batch*(N+1)+current+1
    bonus=tl.load(Bonuses+row)
    tl.store(Lengths+batch,length)
    tl.store(Bonus+batch,bonus)
    tl.store(Probability+batch,tl.load(Residual+row*V+bonus))


def forward(tokens,parents,target,draft,slots,uniforms,bonus_uniforms):
    b,n=tokens.shape;v=target.shape[-1];tiles=triton.cdiv(v,1024)
    residual=torch.empty_like(target)
    scratch=torch.empty((b,v),device=tokens.device,dtype=torch.float32)
    winner=torch.empty((b,n+1),device=tokens.device,dtype=torch.int32)
    bonuses=torch.empty_like(winner)
    mass=torch.empty((b*(n+1),tiles),device=tokens.device,dtype=torch.float64)
    flags=torch.empty((b,),device=tokens.device,dtype=torch.int32)
    paths=torch.empty_like(tokens);compact=torch.empty_like(slots)
    lengths=torch.empty((b,),device=tokens.device,dtype=torch.int32)
    bonus=torch.empty_like(lengths);probability=torch.empty_like(lengths,dtype=torch.float32)
    eagle_sample_init[(b*(n+1),tiles)](target,residual,winner,bonuses,n,v,1024)
    for child in range(n):
        eagle_rejection_decision[(b,)](tokens,parents,draft,residual,uniforms,winner,flags,n,v,child,enable_fp_fusion=False)
        eagle_residual_chunks[(b,tiles)](parents,draft,residual,scratch,mass,n,v,child,tiles,1024)
        eagle_residual_normalize[(b,tiles)](parents,residual,scratch,mass,flags,n,v,child,tiles,triton.next_power_of_2(tiles),1024)
    eagle_probability_masses[(b*(n+1),tiles)](residual,mass,v,tiles,1024)
    eagle_bonus_sample[(b*(n+1),tiles)](residual,mass,bonus_uniforms,bonuses,v,tiles,triton.next_power_of_2(tiles),1024)
    eagle_sampled_walk[(b,)](parents,slots,winner,bonuses,residual,lengths,paths,bonus,compact,probability,n,v,triton.next_power_of_2(n))
    return lengths,paths,bonus,compact,parents,probability
