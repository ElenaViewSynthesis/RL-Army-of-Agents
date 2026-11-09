import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


root = Path(__file__).resolve().parent
evidence = root / "evidence"
project = root.parent.parent
source = project / "artifacts/task-suite-source-20261109T002404Z-stage1/source"


def load(path):
    return json.loads(path.read_text())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)


def positive(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and value > 0, value


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


manifest = load(source / "task-source-manifest.json")
assert load(evidence / "source/task-source-manifest.json") == manifest
for name, digest in manifest["files"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
for name, digest in load(root / "download-receipt.json")["sha256"].items():
    assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest
report = load(evidence / "suite/suite.json")
summary = load(evidence / "summary.json")
request = load(evidence / "request.json")
assert report == load(root / "suite.json")
assert summary["status"] == report["status"] == "pass"
assert summary["source_mismatches"] == []
assert summary["execution"]["returncode"] == 0 and not summary["execution"]["timed_out"]
assert report["counts"] == {"passed": 24, "failed": 0, "unavailable": 0}
assert request["tasks"] == ["gemm_div_sum_scale", "row_softmax", "masked_mean", "prefix_sum"]
assert request["case_set"] == "smoke" and request["gpu_requested"] == "H200"
runtime = json.loads((evidence / "runtime.stdout.log").read_text().strip().splitlines()[-1])
assert runtime["runtime/gpu0_name"] == "NVIDIA H200" and runtime["runtime/gpu0_compute_cap"] == "9.0"
reward = module("_measured_source_reward", source / "utils/reward.py")
catalog = module("_measured_source_catalog", source / "utils/task_catalog.py")
seen = set()
measured = []
for row in report["rows"]:
    task, case, backend, variant = (row["suite/" + k] for k in ("task", "case", "backend", "fixture"))
    key = (task, case, backend, variant)
    assert key not in seen
    seen.add(key)
    path = evidence / "suite" / task / case / backend / variant
    payload = load(path / "suite-case.json")
    assert payload["result"] == row and row["suite/status"] == "passed"
    cfg = payload["config"]
    assert cfg["iters"] == 10 and cfg["speedup_margin"] == .05 and cfg["provenance_threshold"] == .9
    assert "exact_integer_outputs" not in cfg
    for name, digest in payload["protected_sha256"].items():
        relative = f"tasks/{task}/{name}"
        expected = manifest["files"].get(relative, manifest["files"].get(name))
        assert digest == expected, relative
    filename = f"model_new_{backend if variant == 'honest' else variant}.py"
    text = (source / "tasks" / task / filename).read_text()
    if variant != "honest":
        assert text.count('BACKEND = "triton"') == 1
        text = text.replace('BACKEND = "triton"', f"BACKEND = {backend!r}", 1)
    candidate = hashlib.sha256(text.encode()).hexdigest()
    assert payload["candidate_sha256"] == payload["candidate_sha256_after"] == candidate
    if variant == "wrong":
        assert row["env/verify_ok"] is False and row["reward/reason"] == "verification"
        assert row["reward/value"] == -1 and payload["profile"] == {}
        continue
    profile = payload["profile"]
    assert row["env/verify_ok"] is True
    assert profile["timing_measured"] is True and profile["timing_reason"] is None
    assert profile["profile/timing_method"] == "cuda-leaf-sum-two-empty-prime-v1"
    assert profile["profile/diagnostic"] is False and profile["profile/order"] == "gen,eager,compile"
    assert json.loads(profile["profile/primer_event_counts"]) == [0, 0]
    for kind, field in (("gen", "t_gen_us"), ("eager", "t_eager_us"), ("compile", "t_compile_us")):
        samples = json.loads(profile[f"profile/{kind}_kernel_samples"])
        assert len(samples) == profile[f"profile/{kind}_events"] and samples
        for name, value in samples:
            assert isinstance(name, str) and name
            positive(value)
        close(sum(value for name, value in samples) / cfg["iters"], profile[field])
        close(profile[field], row["env/" + field])
    shares = [profile["provenance/" + name] for name in ("agent", "library", "torch", "memory")]
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in shares)
    close(sum(shares), 1.)
    authored = shares[0] / sum(shares[:3])
    close(authored, row["task/provenance_authored"])
    violated = shares[1] > 0 or authored < .9
    expected = reward.compute_reward(
        True, profile["t_eager_us"], profile["t_compile_us"], profile["t_gen_us"],
        0. if violated else authored, SimpleNamespace(**cfg), turns_used=1)
    assert row["reward/value"] == expected["reward"] and row["reward/base"] == expected["reward_base"]
    assert row["reward/reason"] == expected["reason"]
    if variant == "hack":
        assert violated and row["reward/value"] == -1 and row["reward/reason"] == "provenance"
        assert not cfg["benchmark"]
        continue
    assert not violated and authored == 1. and row["task/policy_violation"] is None
    samples_files = list((path / "benchmark-samples").glob("*.json"))
    assert len(samples_files) == 1
    bench = load(samples_files[0])
    assert bench["command_args"]["cold_start"] is True
    assert bench["command_args"]["rounds"] == 6 and bench["command_args"]["iters_per_round"] == 10
    assert bench["summary"]["bench/device_name"] == runtime["runtime/gpu0_name"]
    assert bench["summary"]["bench/device_uuid"] == runtime["runtime/gpu0_uuid"].removeprefix("GPU-")
    for name, value in bench["summary"].items():
        assert row[name] == value
    for kind in ("eager", "compile", "generated"):
        raw = bench["raw"]["samples"][kind]
        for clock in ("event_us", "wall_us"):
            assert len(raw[clock]) == 6
            for value in raw[clock]:
                positive(value)
            close(statistics.median(raw[clock]), row[f"bench/{kind}/{clock}_median"])
        for clock in ("cold_event_us", "cold_wall_us"):
            positive(raw[clock])
            close(raw[clock], row[f"bench/{kind}/{clock}"])
    description = catalog.workload(task, catalog.get_case(task, case))
    latency = row["bench/generated/event_us_median"]
    for key in ("flops", "logical_io_bytes"):
        assert row["bench/workload/" + key] == description[key]
    if description["flops"] is None:
        assert row["bench/generated/tflops"] is None
    else:
        close(row["bench/generated/tflops"], description["flops"] / (latency * 1e6))
    if description["logical_io_bytes"] is None:
        assert row["bench/generated/logical_gbytes_per_s"] is None
    else:
        close(row["bench/generated/logical_gbytes_per_s"],
              description["logical_io_bytes"] / (latency * 1e3))
    measured.append(key)
assert len(seen) == 24 and len(measured) == 8
review = {
    "status": "pass", "run_id": request["run_id"], "rows": report["rows"],
    "checks": {"source_manifest_and_175_source_hashes": True, "downloaded_text_bytes": True,
               "all_24_expected_outcomes": True, "candidate_and_protected_hashes": True,
               "all_16_profiles_raw_sums_and_conditioning": True,
               "all_8_honest_benchmarks_raw_medians_and_cold_samples": True,
               "rewards_reproduced_from_own_profile_only": True},
    "evidence": {"command": summary["execution"]["command"], "run_id": request["run_id"],
                 "function_call_id": "fc-01M6YWEFVW519SS7B542TY90JH",
                 "builder_requested": "2025.06", "runtime": runtime,
                 "source_snapshot": str(source.relative_to(project)),
                 "provenance_source_sha256": manifest["files"]["utils/provenance.py"],
                 "policy_source_sha256": manifest["files"]["utils/task_policy.py"]},
    "limitations": ["Smoke case only, not full grids or flagship performance gates",
                    "Integer outputs used the stock 1e-2 comparison in this historical run",
                    "New exact-integer policy requires a new GPU run",
                    "Benchmark samples and rates are diagnostics, not reward inputs",
                    "No CUTLASS or FlashAttention ceiling was measured"],
}
(root / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": "pass", "verified_expected_cells": 24,
                  "measured_profiles": 16, "honest_benchmarks": 8}))
