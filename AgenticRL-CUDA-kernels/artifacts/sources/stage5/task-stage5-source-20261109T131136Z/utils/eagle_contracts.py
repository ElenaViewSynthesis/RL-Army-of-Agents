# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Documentation-derived evaluation adaptation, 2026-11-09; authored with Codex.
# See legal/eagle4/ and EAGLE4_STAGE5.md.
"""Torch-free Stage 5 grids and algorithmic accounting, derived from docs only.

Eagle 4 Source-Available License; see legal/eagle4/ and EAGLE4_STAGE5.md.
These are repository-defined evaluation contracts, not engine ABI replicas.
"""

TREE_TASKS = (
    "eagle_tree_step", "eagle_tree_row_attention", "eagle_tree_verify_attention",
    "eagle_gdn_tree_conv", "eagle_gdn_tree_recurrence", "eagle_accept_tail",
    "eagle_accept_tail_sampled", "eagle_mamba2_tree_conv",
)


def tree_parents(n, shape):
    if not 1 <= n <= 64 or shape not in ("chain", "wide", "balanced", "ragged"):
        raise ValueError("Tree must have 1..64 nodes and a known shape")
    parents = [-1]
    for i in range(1, n):
        parents.append(i - 1 if shape == "chain" else 0 if shape == "wide"
                       else (i - 1) // 2 if shape == "balanced" else ((i * 1103515245 + 12345) >> 8) % i)
    return parents


def ancestor_rows(parents):
    rows = []
    for node in range(len(parents)):
        path, current = [], node
        while current >= 0:
            path.append(current)
            current = parents[current]
        rows.append(path + [-1] * (len(parents) - len(path)))
    return rows


def stage5_tasks(Task, case):
    trees = tuple((b, n, shape) for b in (1, 8, 32) for n in (1, 4, 8, 16, 32, 64)
                  for shape in ("chain", "wide", "balanced", "ragged"))
    tasks = {}
    def add(name, description, cases):
        tasks[name] = Task(name, 5, description, tuple(cases), ("eagle4",), eval_only=True)
    add("eagle_candidate_topk", "Vocabulary shard top-k with lower-ID ties, then selected-row fp32 gather-dot re-scoring",
        (case("bfloat16", b=b, h=4096, v=v, top_k=k) for b in (1, 8, 32)
         for v in (32000, 128256, 151936) for k in (8, 16)))
    add("eagle_tree_step", "Top-N parent-plus-child scores, parent indices and self-first ancestor paths of new frontier nodes",
        (case(b=b, n=n, tree=shape, top_k=k) for b, n, shape in trees for k in (4, 8)))
    for name in ("eagle_tree_row_attention", "eagle_tree_verify_attention"):
        add(name, "GQA fp32 softmax over committed KV and self-inclusive tree ancestors; bf16 output plus exact ancestor table",
            (case("bfloat16", b=b, n=n, tree=shape, hkv=8, ratio=r, d=d, s=s,
                  window=min(s, 4096) if name == "eagle_tree_row_attention" else s)
             for b, n, shape in trees for r in (1, 4, 8) for d in (64, 128)
             for s in (512, 4096, 32768)))
    for name in ("eagle_gdn_tree_conv", "eagle_mamba2_tree_conv"):
        add(name, "Width-4 tree causal convolution with explicit three-token request history, bias, fp32 output, parent/window tables",
            (case("bfloat16", b=b, n=n, tree=shape, c=c) for b, n, shape in trees
             for c in (192, 384)))
    add("eagle_gdn_tree_recurrence", "Parent-state gated delta update; fp32 per-node outputs and state snapshots, parent/ancestor tables",
        (case(b=b, n=n, tree=shape, heads=h, dk=d, dv=d)
         for b, n, shape in trees for h in (1, 4) for d in (64, 128)))
    add("eagle_accept_tail", "Greedy tree walk from a virtual root; length, path, bonus, compacted slots, parent table; integer exact",
        (case("int32", b=b, n=n, tree=shape) for b, n, shape in trees))
    add("eagle_accept_tail_sampled", "Sequential-child rejection sampling with residual correction, explicit seeded uniforms, and bonus probability",
        (case(b=b, n=n, tree=shape, v=v, seed=17) for b, n, shape in trees
         for v in (32000, 128256, 151936)))
    return tasks


def workload(name, item):
    p = item.params
    b = p["b"]
    if name == "eagle_candidate_topk":
        h, v, k = p["h"], p["v"], p["top_k"]
        flops = 2 * b * h * (v + k)
        io = 2 * (b * h + v * h) + b * k * 12
        convention = "fp32-product vocabulary scores and selected-row re-score, 2 FLOPs per multiply-add; selection comparisons excluded"
    else:
        n = p["n"]
        parents = tree_parents(n, p["tree"])
        depths = [sum(i >= 0 for i in row) for row in ancestor_rows(parents)]
        if name == "eagle_tree_step":
            k = p["top_k"]
            flops = b * n * k
            io = b * (4*n + 4*n*k + 4*n + 4*n + 4*n + 4*n + 4*n*(n+1))
            convention = "one addition per parent-child score; comparisons and ancestor integer work excluded"
        elif name in ("eagle_tree_row_attention", "eagle_tree_verify_attention"):
            hkv, hq, d, s = p["hkv"], p["hkv"]*p["ratio"], p["d"], p["s"]
            visible = sum(depths) + n * p["window"]
            flops = 4 * b * hq * d * visible
            io = 2*b*hq*n*d*2 + 2*b*hkv*(s+n)*d*2 + 4*b*n + 4*b*n*n
            convention = "visible QK and PV only, including self; softmax excluded; IO counts full supplied context even for windowed attention"
        elif name in ("eagle_gdn_tree_conv", "eagle_mamba2_tree_conv"):
            c = p["c"]
            flops = 9*b*n*c
            io = 2*b*(n+3)*c + 4*5*c + 4*b*n*c + 4*b*n*6
            convention = "4 multiply-adds plus bias per channel; negative window entries address request history; integer table work excluded"
        elif name == "eagle_gdn_tree_recurrence":
            h, k, v = p["heads"], p["dk"], p["dv"]
            flops = b*n*h*(7*k*v + 2*v)
            io = 4*b*n*h*(2*k+v+2) + 4*b*h*k*v + 4*b*n*h*(v+k*v) + 4*b*n*(2+n)
            convention = "node walk: decay KV, projection 2KV, residual V, beta scaling V, rank-1 update 2KV, readout 2KV; chain replay recomputation excluded"
        elif name == "eagle_accept_tail":
            flops = 0
            io = 4*b*(7*n+3)
            convention = "integer-only traversal, 0 floating operations; padded outputs fully counted"
        else:
            flops = b * (3*n*p["v"] + (n+1)*(p["v"]-1))
            io = 8*b*(n+1)*p["v"] + 4*b*(8*n+4)
            convention = "all N edges evaluated: V subtracts, V-1 sum adds, V divisions and one ratio per edge; (N+1) length-V CDFs; clamps/comparisons/traversal excluded"
    return {"flops": flops, "logical_io_bytes": io, "flop_convention": convention,
            "bandwidth_convention": "logical input/output bytes once, including metadata and padded outputs; not measured HBM traffic"}
