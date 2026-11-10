"""Torch-free summary statistics and balanced ordering for benchmark samples."""

import math
import statistics


def summarize(samples, prefix=""):
    """Flat n/median/min/max/p25/p75/cv for finite, non-negative samples."""
    values = list(samples)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("samples must be non-empty, finite and non-negative")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    quartiles = (statistics.quantiles(values, n=4, method="inclusive")
                 if len(values) > 1 else [values[0]] * 3)
    return {f"{prefix}n": len(values), f"{prefix}median": statistics.median(values),
            f"{prefix}min": min(values), f"{prefix}max": max(values),
            f"{prefix}p25": quartiles[0], f"{prefix}p75": quartiles[2],
            f"{prefix}cv": std / mean if mean > 0 else 0.0}


def round_order(names, round_index):
    """Rotate left by round_index so each impl leads equally often."""
    names = tuple(names)
    shift = round_index % len(names)
    return names[shift:] + names[:shift]


def _timed_block(entry, impl, kind, rounds, out):
    values = entry[f"{kind}_us"]
    if len(values) != rounds:
        raise ValueError(f"bench/{impl}/{kind}_us must have one sample per round")
    block = summarize(values, prefix=f"bench/{impl}/{kind}_us_")
    out[f"bench/{impl}/{kind}_cv"] = block.pop(f"bench/{impl}/{kind}_us_cv")
    out.update(block)


def summarize_bench(samples, rounds, impls=("eager", "compile", "generated")):
    """Flat bench/* metrics over `rounds` samples; prepare time stays separate."""
    out = {}
    for impl in impls:
        entry = samples[impl]
        for kind in ("event", "wall"):
            _timed_block(entry, impl, kind, rounds, out)
        out[f"bench/{impl}/prepare_s"] = entry["prepare_s"]
        out[f"bench/{impl}/warmup_iters"] = entry["warmup_iters"]
        if "graph_capturable" in entry:
            out[f"bench/{impl}/graph_capturable"] = entry["graph_capturable"]
            out[f"bench/{impl}/graph_error"] = entry["graph_error"]
            out[f"bench/{impl}/graph_replay_ok"] = entry["graph_replay_ok"]
            if entry["graph_us"]:
                _timed_block(entry, impl, "graph", rounds, out)
    out["bench/generated/event_wall_ratio"] = (
        out["bench/generated/event_us_median"] / out["bench/generated/wall_us_median"]
        if out["bench/generated/wall_us_median"] > 0 else None)
    return out
