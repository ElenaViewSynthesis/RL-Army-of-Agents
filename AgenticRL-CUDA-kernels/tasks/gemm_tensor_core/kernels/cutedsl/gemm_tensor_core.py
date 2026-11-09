"""GEMM + fp32 bias in CuTe DSL (CUTLASS 4.8.1 Python DSL). GPU only.

Warp-level MMA path: `cutlass.cute.nvgpu.warp.MmaF16BF16Op` (m16n8k16 f16/bf16
operands, fp32 accumulators) wrapped in a 4-warp `cute.make_tiled_mma`.
A/B tiles are staged through shared memory with universal copy atoms
(predicated for M/N/K edges), `cute.gemm` accumulates, then the fp32 bias is
added and C stored under an identity-tensor bounds predicate covering the
M=8/M=64 tails. Compiled once per (dtype, M, N, K) via `cute.compile` and
launched on the caller's current stream.
"""

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda

_DTYPES = (torch.float16, torch.bfloat16)
BM, BN, BK = 64, 64, 32
THREADS = 128
_COMPILED = {}


@cute.kernel
def gemm_tensor_core_cute_kernel(mA: cute.Tensor, mB: cute.Tensor, bias: cute.Tensor,
                                 mC: cute.Tensor, tiled_mma: cute.TiledMma):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    tileA = cute.local_tile(mA, (BM, BK), (bidy, None))
    tileB = cute.local_tile(mB, (BN, BK), (bidx, None))
    tileC = cute.local_tile(mC, (BM, BN), (bidy, bidx))
    coordsA = cute.local_tile(cute.make_identity_tensor(mA.shape), (BM, BK), (bidy, None))
    coordsB = cute.local_tile(cute.make_identity_tensor(mB.shape), (BN, BK), (bidx, None))
    coordsC = cute.local_tile(cute.make_identity_tensor(mC.shape), (BM, BN), (bidy, bidx))
    smem = cutlass_utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, cute.make_layout((BM, BK), stride=(BK, 1)),
                              byte_alignment=32)
    sB = smem.allocate_tensor(mB.element_type, cute.make_layout((BN, BK), stride=(BK, 1)),
                              byte_alignment=32)
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type,
                                    num_bits_per_copy=16)
    copyA = cute.make_tiled_copy_tv(copy_atom, cute.make_layout((4, 32), stride=(32, 1)),
                                    cute.make_layout((1, 1))).get_slice(tidx)
    copyB = cute.make_tiled_copy_tv(copy_atom, cute.make_layout((32, 4), stride=(1, 32)),
                                    cute.make_layout((1, 1))).get_slice(tidx)
    partAs, partBs = copyA.partition_D(sA), copyB.partition_D(sB)
    predA = cute.make_rmem_tensor(partAs.shape, cutlass.Boolean)
    predB = cute.make_rmem_tensor(partBs.shape, cutlass.Boolean)
    thr_mma = tiled_mma.get_slice(tidx)
    partA, partB = thr_mma.partition_A(sA), thr_mma.partition_B(sB)
    frgA, frgB = thr_mma.make_fragment_A(partA), thr_mma.make_fragment_B(partB)
    partC = thr_mma.partition_C(tileC)
    frgC = thr_mma.make_fragment_C(partC)
    frgC.fill(0.0)
    k_tiles = cute.ceil_div(mA.shape[1], BK)
    for kt in range(0, k_tiles, 1):
        crdA = copyA.partition_S(coordsA[None, None, kt])
        crdB = copyB.partition_S(coordsB[None, None, kt])
        for i in range(0, cute.size(predA), 1):
            predA[i] = cute.elem_less(crdA[i], mA.shape)
        for i in range(0, cute.size(predB), 1):
            predB[i] = cute.elem_less(crdB[i], mB.shape)
        partAs.fill(0)
        partBs.fill(0)
        cute.copy(copy_atom, copyA.partition_S(tileA[None, None, kt]), partAs, pred=predA)
        cute.copy(copy_atom, copyB.partition_S(tileB[None, None, kt]), partBs, pred=predB)
        cute.arch.barrier()
        cute.copy(copy_atom, partA, frgA)
        cute.copy(copy_atom, partB, frgB)
        cute.gemm(tiled_mma, frgC, frgA, frgB, frgC)
        cute.arch.barrier()
    output_coords = thr_mma.partition_C(coordsC)
    for i in range(0, cute.size(frgC), 1):
        coord = output_coords[i]
        if cute.elem_less(coord, mC.shape):
            partC[i] = frgC[i] + bias[coord[1]]


@cute.jit
def gemm_tensor_core_cute(mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor,
                          mC: cute.Tensor, stream: cuda.CUstream):
    ab_dtype = mA.element_type
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, cutlass.Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(mma_op, (2, 2, 1))

    transposed_b = cute.make_tensor(mB.iterator, cute.make_layout(
        (mB.shape[1], mB.shape[0]), stride=(mB.stride[1], mB.stride[0])))
    grid = (cute.ceil_div(mC.shape[1], BN), cute.ceil_div(mC.shape[0], BM), 1)
    gemm_tensor_core_cute_kernel(mA, transposed_b, mBias, mC, tiled_mma).launch(
        grid=grid, block=[THREADS, 1, 1], stream=stream)


def gemm_tensor_core_forward(a, b, bias):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm_tensor_core requires a [M,K] and b [K,N]")
    if a.dtype != b.dtype or a.device != b.device or a.device != bias.device:
        raise ValueError("a, b and bias must share dtype and device")
    if bias.ndim != 1 or bias.shape[0] != b.shape[1] or bias.dtype != torch.float32:
        raise ValueError("bias must be float32 [N]")
    if a.dtype not in _DTYPES:
        raise ValueError("gemm_tensor_core requires float16 or bfloat16 inputs")
    if not a.is_contiguous() or not b.is_contiguous() or not bias.is_contiguous():
        raise ValueError("gemm_tensor_core requires contiguous inputs")
    if a.device.type != "cuda":
        raise ValueError("CuTe DSL gemm requires CUDA tensors (no CPU interpreter)")
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    if m and n:
        with torch.cuda.device(a.device):
            args = (from_dlpack(a), from_dlpack(b), from_dlpack(bias),
                    from_dlpack(out), cutlass_torch.current_stream())
            key = (a.device.index, a.dtype, m, n, k)
            compiled = _COMPILED.get(key)
            if compiled is None:
                compiled = cute.compile(gemm_tensor_core_cute, *args)
                _COMPILED[key] = compiled
            compiled(*args)
    return out
