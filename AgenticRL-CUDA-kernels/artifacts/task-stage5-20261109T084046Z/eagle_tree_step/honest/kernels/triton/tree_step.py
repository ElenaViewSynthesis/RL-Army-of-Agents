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
def eagle_frontier_select(Scores, Children, Parents, Out, ParentIds, Paths, N:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    batch=tl.program_id(0)
    ix=tl.arange(0,BLOCK)
    values=tl.load(Scores+batch*N+ix//K,ix<N*K,other=0)+tl.load(Children+batch*N*K+ix,ix<N*K,other=-float('inf'))
    for rank in range(N):
        best=tl.max(values,0)
        selected=tl.min(tl.where((values==best)&(ix<N*K),ix,2147483647),0)
        parent=selected//K
        tl.store(Out+batch*N+rank,best)
        tl.store(ParentIds+batch*N+rank,parent)
        tl.store(Paths+(batch*N+rank)*(N+1),N+rank)
        current=parent
        for depth in range(N):
            tl.store(Paths+(batch*N+rank)*(N+1)+depth+1,current)
            current=tl.load(Parents+batch*N+tl.maximum(current,0),current>=0,other=-1)
        values=tl.where(ix==selected,-float('inf'),values)


def forward(scores,children,parents):
    b,n,k=children.shape
    output=torch.empty_like(scores);ids=torch.empty_like(parents)
    paths=torch.empty((b,n,n+1),device=scores.device,dtype=torch.int32)
    eagle_frontier_select[(b,)](scores,children,parents,output,ids,paths,n,k,triton.next_power_of_2(n*k))
    return output,ids,paths
