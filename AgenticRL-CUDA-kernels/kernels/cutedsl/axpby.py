# Adapted from NVIDIA CUTLASS examples/python/CuTeDSL/cute/ampere/kernel/
# elementwise/elementwise_add.py: scaled by alpha, viewed as a (1, n) row,
# launched on torch's current stream, and compiled once per dtype.
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Fused alpha * a + b in CuTe DSL (CUTLASS 4.x Python DSL). GPU only.

Each thread block copies a (1, THREADS * vector) tile through registers under
an element-wise bounds predicate, so any length works. The default device
symbol looks like kernel_cutlass_axpby_cute_kernel_..._0; the provenance
profiler credits it by the @cute.kernel function name. Not yet run on hardware.
"""

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
THREADS = 128
_COMPILED = {}


@cute.kernel
def axpby_cute_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, cC: cute.Tensor,
                      shape: cute.Shape, alpha: cutlass.Float32,
                      thr_layout: cute.Layout, val_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA, blkB, blkC, blkCrd = gA[blk_coord], gB[blk_coord], gC[blk_coord], cC[blk_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)
    thr_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout).get_slice(tidx)
    thr_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout).get_slice(tidx)
    thr_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout).get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    # Element-wise bounds predicate from the identity (coordinate) tensor.
    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)
    a = frgA.load().to(cutlass.Float32)
    b = frgB.load().to(cutlass.Float32)
    frgC.store((a * alpha + b).to(gC.element_type))
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def axpby_cute(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, alpha: cutlass.Float32,
               stream: cuda.CUstream, copy_bits: cutlass.Constexpr = 128):
    vector = copy_bits // mA.element_type.width
    thr_layout = cute.make_ordered_layout((1, THREADS), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, vector), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler_mn)

    axpby_cute_kernel(gA, gB, gC, cC, mC.shape, alpha, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )


def _row(x):
    return from_dlpack(x.view(1, -1)).mark_layout_dynamic(leading_dim=1)


def axpby_forward(a, b, alpha):
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("a and b must have the same shape, dtype, and device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("axpby requires contiguous inputs")
    if a.dtype not in _DTYPES:
        raise ValueError("axpby requires floating point inputs")
    if a.device.type != "cuda":
        raise ValueError("CuTe DSL axpby requires CUDA tensors (no CPU interpreter)")
    out = torch.empty_like(a)
    if a.numel():
        # Launch on the caller's current stream (stream/device contract).
        args = (_row(a), _row(b), _row(out), cutlass.Float32(alpha), cutlass_torch.current_stream())
        compiled = _COMPILED.get(a.dtype)
        if compiled is None:
            # Compile once per dtype (dynamic layout covers every length), never per call.
            compiled = _COMPILED[a.dtype] = cute.compile(axpby_cute, *args)
        compiled(*args)
    return out
