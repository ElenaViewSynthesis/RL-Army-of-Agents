"""Approved authored-only gate for new workloads, not the stock rollout API."""

import dataclasses
import math

from utils.reward import compute_reward


POLICY = "authored-only-no-library-v1"
_SCORED = frozenset(("correct", "faster_than_eager", "faster_than_both", "provenance"))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def authored_fraction(profile):
    shares = [profile.get("provenance/" + name) for name in ("agent", "library", "torch", "memory")]
    if (not all(_number(value) and 0 <= value <= 1 for value in shares)
            or not math.isclose(sum(shares), 1.0, rel_tol=1e-6, abs_tol=1e-6)):
        return None
    non_memory = sum(shares[:3])
    return shares[0] / non_memory if non_memory > 0 else None


def apply_task_policy(result, profile, cfg, turns_used=None):
    output = dict(result)
    fraction = authored_fraction(profile)
    strict_cfg = dataclasses.replace(cfg, provenance_threshold=max(.9, cfg.provenance_threshold))
    output.update({"task/provenance_policy": POLICY,
                   "task/provenance_authored": fraction,
                   "task/provenance_threshold": strict_cfg.provenance_threshold,
                   "task/legacy_reward_value": result.get("reward/value"),
                   "task/policy_violation": None})
    if result.get("reward/reason") not in _SCORED:
        return output
    if profile.get("profile/diagnostic"):
        output.update({"reward/value": -1.0, "reward/base": -1,
                       "reward/reason": "diagnostic_profile", "reward/overlong_penalty": 0.0,
                       "task/policy_violation": "diagnostic_profile"})
        return output
    if fraction is None:
        violation = "invalid_or_missing_provenance_shares"
    elif profile["provenance/library"] > 0:
        violation = "library_kernel_time"
    elif fraction < strict_cfg.provenance_threshold:
        violation = "authored_fraction"
    else:
        violation = None
    reward = compute_reward(result["env/verify_ok"], result["env/t_eager_us"],
                            result["env/t_compile_us"], result["env/t_gen_us"],
                            0.0 if violation else fraction, strict_cfg, turns_used)
    output.update({"reward/value": reward["reward"], "reward/base": reward["reward_base"],
                   "reward/reason": reward["reason"],
                   "reward/overlong_penalty": reward["overlong_penalty"],
                   "task/policy_violation": violation})
    return output


def diagnostic_rates(description, latency_us):
    flops, io_bytes = description.get("flops"), description.get("logical_io_bytes")
    valid = _number(latency_us) and latency_us > 0
    return {
        "bench/workload/flops": flops,
        "bench/workload/logical_io_bytes": io_bytes,
        "bench/workload/flop_convention": description.get("flop_convention", "unavailable"),
        "bench/workload/bandwidth_convention": description.get("bandwidth_convention", "unavailable"),
        "bench/generated/tflops": flops / (latency_us * 1e6)
        if valid and _number(flops) and flops >= 0 else None,
        "bench/generated/logical_gbytes_per_s": io_bytes / (latency_us * 1e3)
        if valid and _number(io_bytes) and io_bytes >= 0 else None,
    }


def gemm_ceiling_issue(record, params, dtype):
    if record.get("ceiling/available") is not True:
        return record.get("ceiling/unavailable_reason") or "ceiling_unavailable"
    for key in ("m", "n", "k"):
        value = record.get("ceiling/" + key)
        if type(value) is not int or value != params[key]:
            return "ceiling_shape_mismatch"
    if record.get("ceiling/dtype") != "torch." + dtype:
        return "ceiling_dtype_mismatch"
    if str(record.get("ceiling/provider", "")).split(".")[-1].lower() != "cublas":
        return "ceiling_provider_not_cublas"
    for key, expected in (("warmup_iters", 10), ("rounds", 6), ("iters_per_round", 10)):
        if record.get("ceiling/" + key) != expected:
            return "ceiling_sampling_mismatch"
    flops = 2 * params["m"] * params["n"] * params["k"]
    if record.get("ceiling/flops") != flops:
        return "ceiling_flop_count_mismatch"
    cublas = record.get("ceiling/cublas_event_us_median")
    generated = record.get("ceiling/generated_event_us_median")
    if not all(_number(value) and value > 0 for value in (cublas, generated)):
        return "ceiling_invalid_latency"
    expected = {"ceiling/cublas_fraction": cublas / generated,
                "ceiling/cublas_tflops": flops / (cublas * 1e6),
                "ceiling/generated_tflops": flops / (generated * 1e6)}
    if any(not _number(record.get(key)) or
           not math.isclose(record[key], value, rel_tol=1e-9, abs_tol=0)
           for key, value in expected.items()):
        return "ceiling_inconsistent_derived_rate"
    return None
