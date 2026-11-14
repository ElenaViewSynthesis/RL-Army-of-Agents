"""Run one task's CUDA-graph capture and replay under NVTX markers (diagnostic only).

Runs inside a task workdir through rollout_env.bootstrap_script, normally under
`nsys profile --cuda-graph-trace=node` (see scripts/graph_trace.py). It mirrors
utils.benchmark._capture_graph: side-stream warm-up x3, capture on static input
clones, copy fresh inputs, replay, compare with the reference. Each phase is
an NVTX range so the timeline can be split by phase. Nothing here feeds a
reward; the result is a JSON record of what was captured and whether replay
matched the reference.
"""

import argparse
import json
import os
import time
from pathlib import Path

IMPLS = ("generated", "eager")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--eager-forwards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--impls", default=",".join(IMPLS))
    args = parser.parse_args(argv)
    if args.replays <= 0 or args.eager_forwards < 0:
        parser.error("--replays must be positive and --eager-forwards non-negative")
    args.impls = [name for name in args.impls.split(",") if name]
    if not args.impls or set(args.impls) - set(IMPLS):
        parser.error(f"--impls must be a nonempty subset of {','.join(IMPLS)}")
    return args


def range_name(impl, phase):
    """The NVTX range label; scripts/render_nsys_timeline.py parses it back."""
    return f"{impl}: {phase}"


def main(argv=None):
    args = parse_args(argv)
    import torch
    from utils.verification import build_inputs, check_equal, initialize_models

    compare = check_equal
    if os.environ.get("CUDA_AGENT_RELATIVE_ERROR_BOUNDS") == "1":
        from utils.lora_verification import check_outputs
        compare = lambda actual, expected, fresh: check_outputs(  # noqa: E731
            os.environ["CUDA_AGENT_TASK"], actual, expected, fresh)
    else:
        base = compare
        compare = lambda actual, expected, fresh: base(actual, expected)  # noqa: E731

    torch.manual_seed(args.seed)
    torch_model, cuda_model = initialize_models()
    torch_inputs, cuda_inputs = build_inputs()
    models = {"generated": (cuda_model, cuda_inputs), "eager": (torch_model, torch_inputs)}
    nvtx = torch.cuda.nvtx
    record = {"schema": "graph-trace-worker-v1", "task": os.environ.get("CUDA_AGENT_TASK"),
              "case": os.environ.get("CUDA_AGENT_TASK_CASE"), "replays": args.replays,
              "eager_forwards": args.eager_forwards, "seed": args.seed, "impls": {}}

    def phase(impl, name, call):
        nvtx.range_push(range_name(impl, name))
        t0 = time.perf_counter()
        try:
            return call()
        finally:
            torch.cuda.synchronize()
            nvtx.range_pop()
            timings.setdefault(name, []).append((time.perf_counter() - t0) * 1e6)

    with torch.no_grad():
        for impl in args.impls:
            model, inputs = models[impl]
            timings = {}
            entry = {"graph_capturable": False, "graph_replay_ok": None, "graph_error": None,
                     "host_us": timings}
            # JIT/compile and lazy initialization stay outside every NVTX range.
            model(*inputs)
            torch.cuda.synchronize()
            for _ in range(args.eager_forwards):
                phase(impl, "forward (no graph)", lambda: model(*inputs))
            try:
                static_inputs = [t.clone() for t in inputs]
                side = torch.cuda.Stream()

                def warmup():
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            model(*static_inputs)
                    torch.cuda.current_stream().wait_stream(side)

                phase(impl, "warm-up (side stream x3)", warmup)
                graph = torch.cuda.CUDAGraph()

                def capture():
                    with torch.cuda.graph(graph):
                        return model(*static_inputs)

                static_out = phase(impl, "capture", capture)
                entry["graph_capturable"] = True
                fresh_torch, fresh_cuda = build_inputs()
                fresh = fresh_cuda if impl == "generated" else fresh_torch
                for index in range(args.replays):
                    def replay():
                        for dst, src in zip(static_inputs, fresh):
                            dst.copy_(src)
                        graph.replay()
                    phase(impl, f"replay {index + 1}", replay)
                try:
                    compare(static_out, torch_model(*fresh), fresh)
                    entry["graph_replay_ok"] = True
                except Exception as exc:  # replay mismatch is data, not a crash
                    entry["graph_replay_ok"] = False
                    entry["graph_error"] = f"{type(exc).__name__}: {exc}"[:512]
            except Exception as exc:
                entry["graph_error"] = f"{type(exc).__name__}: {exc}"[:512]
            record["impls"][impl] = entry
    index = torch.cuda.current_device()
    record["device_name"] = torch.cuda.get_device_name(index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "impls"}))


if __name__ == "__main__":
    main()
