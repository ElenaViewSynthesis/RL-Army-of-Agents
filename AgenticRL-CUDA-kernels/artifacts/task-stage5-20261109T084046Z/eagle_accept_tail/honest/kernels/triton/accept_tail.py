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
def eagle_greedy_walk(Tokens,Parents,Targets,Slots,Lengths,Paths,Bonus,Compact,N:tl.constexpr,BLOCK:tl.constexpr):
    batch=tl.program_id(0)
    ix=tl.arange(0,BLOCK)
    parents=tl.load(Parents+batch*N+ix,ix<N,other=-2)
    tokens=tl.load(Tokens+batch*N+ix,ix<N,other=-1)
    tl.store(Paths+batch*N+ix,-1,ix<N)
    tl.store(Compact+batch*N+ix,-1,ix<N)
    current=-1;length=0;active=1
    for _ in range(N):
        target=tl.load(Targets+batch*(N+1)+current+1)
        selected=tl.min(tl.where((parents==current)&(tokens==target)&(ix<N),ix,N),0)
        if (active==1)&(selected<N):
            slot=tl.load(Slots+batch*N+selected)
            tl.store(Paths+batch*N+length,selected)
            tl.store(Compact+batch*N+length,slot)
            current=selected;length+=1
        else:
            active=0
    tl.store(Lengths+batch,length)
    tl.store(Bonus+batch,tl.load(Targets+batch*(N+1)+current+1))


def forward(tokens,parents,targets,slots):
    b,n=tokens.shape
    paths=torch.empty_like(tokens);compact=torch.empty_like(slots)
    lengths=torch.empty((b,),device=tokens.device,dtype=torch.int32)
    bonus=torch.empty_like(lengths)
    eagle_greedy_walk[(b,)](tokens,parents,targets,slots,lengths,paths,bonus,compact,n,triton.next_power_of_2(n))
    return lengths,paths,bonus,compact,parents
