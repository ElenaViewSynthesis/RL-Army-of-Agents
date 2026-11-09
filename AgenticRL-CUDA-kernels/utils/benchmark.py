"""Repeatable latency benchmark, separate from the reward profiler pass.

Stream/device contract: ModelNew.forward must enqueue all device work on
torch.cuda.current_stream() of the inputs' device, or make that stream
wait on side streams before returning. Event latency covers only that
stream; wall latency brackets torch.cuda.synchronize() and includes CPU
dispatch and every stream. bench/generated/event_wall_ratio exposes
un-joined side-stream work. Prepare (compile/JIT/autotune) and warmup are
timed separately and never enter the steady-state statistics. All output
is diagnostics; the reward metric (t_*_us) comes from utils.profiling.
"""

import argparse
import json
import os
import time
from pathlib import Path

from utils import stats
from utils.backend import BACKEND_CHOICES, DEVICE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the flat JSON record")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=6,
                        help="Must be a positive multiple of 3")
    parser.add_argument("--iters-per-round", type=int, default=10)
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Also capture and time CUDA-graph replays")
    parser.add_argument("--samples-json", type=Path,
                        help="Also write the summary and raw samples to this file")
    parser.add_argument("--cold-start", action="store_true",
                        default=os.environ.get("CUDA_AGENT_BENCH_COLD") == "1",
                        help="Record the first post-initialization call separately; no cache flush")
    args = parser.parse_args(argv)
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")
    if args.rounds <= 0 or args.rounds % 3:
        parser.error("--rounds must be a positive multiple of 3")
    if args.iters_per_round <= 0:
        parser.error("--iters-per-round must be positive")
    return args


def unavailable_reason():
    """Torch-free availability check; returns a reason string or None."""
    if DEVICE.split(":")[0] == "cpu":
        return "CUDA_AGENT_DEVICE=cpu"
    return None


def _time_calls(call, iters):
    """One measured round: CUDA events plus a synchronized wall bracket."""
    import torch

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - t0) * 1e6
    return start.elapsed_time(end) * 1000 / iters, wall_us / iters


def _capture_graph(model, inputs, fresh, torch, torch_model, check_equal):
    """Side-stream warmup then capture; failure is data, never a crash."""
    entry = {"graph_capturable": False, "graph_error": None,
             "graph_replay_ok": None, "graph_us": []}
    try:
        static_inputs = [t.clone() for t in inputs]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                model(*static_inputs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = model(*static_inputs)
        entry["graph_capturable"] = True
        try:
            for dst, src in zip(static_inputs, fresh):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            check_equal(static_out, torch_model(*fresh))
            entry["graph_replay_ok"] = True
        except Exception:
            entry["graph_replay_ok"] = False
        return entry, graph
    except Exception as exc:
        entry["graph_error"] = f"{type(exc).__name__}: {exc}"[:512]
        return entry, None


def run_benchmark(args):
    """Return (flat record, raw samples/order). Timings are diagnostics only."""
    reason = unavailable_reason()
    if reason is None:
        import torch
        if not torch.cuda.is_available():
            reason = "torch.cuda.is_available() is False"
    if reason is not None:
        return {"bench/available": False, "bench/unavailable_reason": reason}, {}
    from utils.verification import build_inputs, check_equal, initialize_models

    torch.manual_seed(args.seed)
    torch_model, cuda_model = initialize_models()
    torch_inputs, cuda_inputs = build_inputs()
    impls = ("eager", "compile", "generated")
    callables = {"eager": (torch_model, torch_inputs),
                 "compile": (torch.compile(torch_model), torch_inputs),
                 "generated": (cuda_model, cuda_inputs)}
    samples = {name: {"event_us": [], "wall_us": [], "prepare_s": None,
                      "warmup_iters": args.warmup_iters} for name in impls}
    graphs = {}
    order = []
    with torch.no_grad():
        # Prepare and warm up in a fixed order; prepare time is not a sample.
        for name in impls:
            model, inputs = callables[name]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if getattr(args, "cold_start", False):
                event_us, wall_us = _time_calls(lambda: model(*inputs), 1)
                samples[name].update(cold_event_us=event_us, cold_wall_us=wall_us)
            else:
                model(*inputs)
                torch.cuda.synchronize()
            samples[name]["prepare_s"] = time.perf_counter() - t0
            for _ in range(args.warmup_iters):
                model(*inputs)
            torch.cuda.synchronize()
            if args.cuda_graphs:
                fresh_torch, fresh_cuda = build_inputs()
                entry, graph = _capture_graph(
                    model, inputs, fresh_cuda if name == "generated" else fresh_torch,
                    torch, torch_model, check_equal)
                samples[name].update(entry)
                graphs[name] = graph
        # Balanced rotation: each impl occupies each position equally often.
        for r in range(args.rounds):
            rotated = stats.round_order(impls, r)
            order.append(",".join(rotated))
            for name in rotated:
                model, inputs = callables[name]
                event_us, wall_us = _time_calls(lambda: model(*inputs), args.iters_per_round)
                samples[name]["event_us"].append(event_us)
                samples[name]["wall_us"].append(wall_us)
                if args.cuda_graphs and graphs.get(name) is not None:
                    graph = graphs[name]
                    event_us, _ = _time_calls(graph.replay, args.iters_per_round)
                    samples[name]["graph_us"].append(event_us)
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    record = {"bench/available": True,
              "bench/device_index": index,
              "bench/device_name": torch.cuda.get_device_name(index),
              "bench/device_capability": f"{props.major}.{props.minor}",
              "bench/device_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
              "bench/warmup_iters": args.warmup_iters,
              "bench/rounds": args.rounds,
              "bench/iters_per_round": args.iters_per_round,
              "bench/seed": args.seed,
              "bench/order": "|".join(order)}
    record.update(stats.summarize_bench(samples, args.rounds))
    if getattr(args, "cold_start", False):
        record["bench/cold_definition"] = (
            "first forward after initialize_models in this process; context initialized; "
            "includes lazy compilation; not cache-flushed")
        for name in impls:
            for kind in ("event", "wall"):
                record[f"bench/{name}/cold_{kind}_us"] = samples[name][f"cold_{kind}_us"]
    if args.cuda_graphs:
        record["bench/graph_all_capturable"] = all(
            s["graph_capturable"] for s in samples.values())
    raw = {"samples": samples, "order": order}
    return record, raw


def main():
    args = parse_args()
    record, raw = run_benchmark(args)
    # Opt-in artifact export; raw samples never enter the reward or flat metrics.
    import os
    payload = json.dumps({"summary": record, "raw": raw,
                          "command_args": {k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(args).items()}},
                         indent=2, allow_nan=False) + "\n"
    directory = os.environ.get("CUDA_AGENT_BENCH_SAMPLES_DIR")
    if directory:
        from uuid import uuid4
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / (uuid4().hex + ".json")).write_text(payload)
    if args.samples_json:
        args.samples_json.parent.mkdir(parents=True, exist_ok=True)
        args.samples_json.write_text(payload)
    if args.json:
        print(json.dumps(record, allow_nan=False))
    elif record["bench/available"]:
        for impl in ("eager", "compile", "generated"):
            print(f"{impl}: event median {record[f'bench/{impl}/event_us_median']:.3f}us, "
                  f"wall median {record[f'bench/{impl}/wall_us_median']:.3f}us "
                  f"(n={record[f'bench/{impl}/event_us_n']})")
    else:
        print(f"[SKIP] benchmark unavailable: {record['bench/unavailable_reason']}")


if __name__ == "__main__":
    main()
