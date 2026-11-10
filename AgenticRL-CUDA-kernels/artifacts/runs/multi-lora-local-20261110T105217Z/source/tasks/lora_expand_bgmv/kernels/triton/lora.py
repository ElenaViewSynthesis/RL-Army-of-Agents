"""Authored multi-LoRA kernels. Fixed tiles, fp32 arithmetic, no autotuning."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def lora_initialize(Src, Out, N: tl.constexpr, COPY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    if COPY:
        value = tl.load(Src+i, i < N, 0)
    tl.store(Out+i, value, i < N)


@tr.jit
def lora_count_slots(Idx, Counts, T: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr):
    slot = tl.program_id(0)
    t = tl.arange(0, BLOCK)
    ids = tl.load(Idx+t, t < T, -2)
    key = tl.where(ids < 0, L, ids)
    tl.store(Counts+slot, tl.sum(((key == slot) & (t < T)).to(tl.int32), 0))


@tr.jit
def lora_group_tokens(Idx, Counts, Order, Ids, Starts, Lengths, Active,
                      T: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr, LS: tl.constexpr):
    slot = tl.program_id(0)
    slots = tl.arange(0, LS)
    counts = tl.load(Counts+slots, slots <= L, 0)
    start = tl.sum(tl.where(slots < slot, counts, 0), 0)
    segment = tl.sum(((slots < slot) & (counts > 0)).to(tl.int32), 0)
    count = tl.load(Counts+slot)
    if slot < L and count > 0:
        tl.store(Ids+segment, slot)
        tl.store(Starts+segment, start)
        tl.store(Lengths+segment, count)
    active = tl.sum(((slots < L) & (counts > 0)).to(tl.int32), 0)
    if slot == L:
        tl.store(Active, active)
        # Each inactive segment gets a canonical value, no overlapping writers.
        tl.store(Ids+slots, -1, (slots >= active) & (slots < L))
        tl.store(Starts+slots, 0, (slots >= active) & (slots < L))
        tl.store(Lengths+slots, 0, (slots >= active) & (slots < L))
    t = tl.arange(0, BLOCK)
    idx = tl.load(Idx+t, t < T, -2)
    selected = (tl.where(idx < 0, L, idx) == slot) & (t < T)
    offset = tl.cumsum(selected.to(tl.int32), 0)-1
    tl.store(Order+start+offset, t, selected)


def metadata(idx, slots):
    t, l = idx.numel(), slots.shape[0]
    counts = torch.empty((l+1,), device=idx.device, dtype=torch.int32)
    order = torch.empty_like(idx)
    ids, starts, lengths = [torch.empty((l,), device=idx.device, dtype=torch.int32) for _ in range(3)]
    active = torch.empty((1,), device=idx.device, dtype=torch.int32)
    lora_count_slots[(l+1,)](idx, counts, t, l, tr.next_power_of_2(t))
    lora_group_tokens[(l+1,)](idx, counts, order, ids, starts, lengths, active,
                             t, l, tr.next_power_of_2(t), tr.next_power_of_2(l+1))
    return order, ids, starts, lengths, active


@tr.jit
def lora_decode_shrink(X, A, Idx, Ranks, Scales, Scratch,
                       T: tl.constexpr, H: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                       SPLITS: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    rank, token, sl = row % R, row // R % T, row // (R*T)
    slot = tl.load(Idx+token).to(tl.int64)
    value = tl.full((), 0, tl.float32)
    if slot >= 0:
        limit = tl.load(Ranks+slot)
        if rank < limit:
            k = split*K + tl.arange(0, K)
            x = tl.load(X+token*H+k, k < H, 0).to(tl.float32)
            a = tl.load(A+((slot*S+sl)*R+rank)*H+k, k < H, 0).to(tl.float32)
            value = tl.sum(x*a, 0)*tl.load(Scales+slot)
    tl.store(Scratch+row*SPLITS+split, value)


@tr.jit
def lora_merge_shrink(Scratch, Buf, N: tl.constexpr, SPLITS: tl.constexpr,
                      BS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    split = tl.arange(0, BS)
    value = tl.load(Scratch+row[:, None]*SPLITS+split[None, :],
                    (row[:, None] < N) & (split[None, :] < SPLITS), 0)
    tl.store(Buf+row, tl.sum(value, 1), row < N)


@tr.jit
def lora_decode_expand(Buf, B, Y, Idx, Ranks, Offsets, Out,
                       T: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
                       S: tl.constexpr, W: tl.constexpr, RN: tl.constexpr, BN: tl.constexpr):
    tile = tl.program_id(0).to(tl.int64)
    tiles: tl.constexpr = tr.cdiv(N, BN)
    token, sl = tile // tiles % T, tile // (tiles*T)
    slot = tl.load(Idx+token).to(tl.int64)
    if slot >= 0:
        n = tile % tiles*BN + tl.arange(0, BN)
        r = tl.arange(0, RN)
        limit = tl.load(Ranks+slot)
        b = tl.load(B+((slot*S+sl)*N+n[:, None])*R+r[None, :],
                    (n[:, None] < N) & (r[None, :] < limit), 0).to(tl.float32)
        buf = tl.load(Buf+(sl*T+token)*R+r, r < limit, 0)
        delta = tl.sum(b*buf[None, :], 1)
        offset = tl.load(Offsets+sl).to(tl.int64)
        old = tl.load(Y+token*W+offset+n, n < N, 0).to(tl.float32)
        tl.store(Out+token*W+offset+n, old+delta, n < N)


@tr.jit
def lora_segment_gemm(X, Weight, Y, Ranks, Scales, Offsets, Order, Ids, Starts, Lengths, Active, Out,
                      T: tl.constexpr, H: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                      N: tl.constexpr, W: tl.constexpr, L: tl.constexpr,
                      SHRINK: tl.constexpr, M: tl.constexpr=16, BN: tl.constexpr=16, BK: tl.constexpr=32):
    pid = tl.program_id(0).to(tl.int64)
    NT: tl.constexpr = tr.cdiv(R if SHRINK else N, BN)
    MT: tl.constexpr = tr.cdiv(T, M)
    nt, mt = pid % NT, pid // NT % MT
    seg, sl = pid // (NT*MT) % L, pid // (NT*MT*L)
    if seg < tl.load(Active):
        length = tl.load(Lengths+seg)
        if mt*M < length:
            slot = tl.load(Ids+seg).to(tl.int64)
            rank = tl.load(Ranks+slot)
            m = mt*M+tl.arange(0, M)
            token = tl.load(Order+tl.load(Starts+seg)+m, m < length, 0).to(tl.int64)
            n = nt*BN+tl.arange(0, BN)
            kk = tl.arange(0, BK)
            acc = tl.full((M, BN), 0, tl.float32)
            for base in range(tr.cdiv(H if SHRINK else R, BK)):
                k = base*BK+kk
                if SHRINK:
                    xv = tl.load(X+token[:, None]*H+k[None, :], (m[:, None] < length) & (k[None, :] < H), 0).to(tl.float32)
                    wv = tl.load(Weight+((slot*S+sl)*R+n[None, :])*H+k[:, None],
                                 (n[None, :] < rank) & (k[:, None] < H), 0).to(tl.float32)
                else:
                    xv = tl.load(X+(sl*T+token[:, None])*R+k[None, :],
                                 (m[:, None] < length) & (k[None, :] < rank), 0)
                    wv = tl.load(Weight+((slot*S+sl)*N+n[None, :])*R+k[:, None],
                                 (n[None, :] < N) & (k[:, None] < rank), 0).to(tl.float32)
                acc += tl.dot(xv, wv, input_precision="ieee")
            if SHRINK:
                acc *= tl.load(Scales+slot)
                tl.store(Out+(sl*T+token[:, None])*R+n[None, :], acc,
                         (m[:, None] < length) & (n[None, :] < rank))
            else:
                dest = token[:, None]*W+tl.load(Offsets+sl).to(tl.int64)+n[None, :]
                old = tl.load(Y+dest, (m[:, None] < length) & (n[None, :] < N), 0).to(tl.float32)
                tl.store(Out+dest, old+acc, (m[:, None] < length) & (n[None, :] < N))


def shrink(x, a, idx, ranks, scales, *meta):
    l, s, r, h = a.shape
    t = x.shape[0]
    out = torch.empty((s, t, r), dtype=torch.float32, device=x.device)
    if meta:
        lora_initialize[(tr.cdiv(out.numel(), 256),)](out, out, out.numel(), False, 256)
        grid = s*l*tr.cdiv(t, 16)*tr.cdiv(r, 16)
        lora_segment_gemm[(grid,)](x, a, out, ranks, scales, idx, *meta, out,
                                    t, h, r, s, h, 0, l, True)
    else:
        splits = tr.cdiv(h, 1024)
        scratch = torch.empty((s*t*r, splits), device=x.device, dtype=torch.float32)
        lora_decode_shrink[(s*t*r, splits)](x, a, idx, ranks, scales, scratch, t, h, r, s, splits, 1024)
        lora_merge_shrink[(tr.cdiv(s*t*r, 64),)](scratch, out, s*t*r, splits, tr.next_power_of_2(splits), 64)
    return out


def expand(buf, b, y, idx, ranks, offsets, *meta):
    l, s, n, r = b.shape
    t, w = y.shape
    out = torch.empty_like(y)
    lora_initialize[(tr.cdiv(y.numel(), 256),)](y, out, y.numel(), True, 256)
    if meta:
        grid = s*l*tr.cdiv(t, 16)*tr.cdiv(n, 16)
        lora_segment_gemm[(grid,)](buf, b, y, ranks, idx, offsets, *meta, out,
                                    t, 0, r, s, n, w, l, False)
    else:
        lora_decode_expand[(s*t*tr.cdiv(n, 16),)](buf, b, y, idx, ranks, offsets, out,
                                                t, n, r, s, w, tr.next_power_of_2(r), 16)
    return out


def forward(name, *args):
    if name == "lora_segment_metadata":
        return metadata(*args)
    if "fused" in name:
        x, a, b, y, idx, ranks, scales, offsets, *meta = args
        buf = shrink(x, a, idx, ranks, scales, *meta)
        return expand(buf, b, y, idx, ranks, offsets, *meta)
    return shrink(*args) if "shrink" in name else expand(*args)
