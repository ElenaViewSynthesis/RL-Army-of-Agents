"""Synthetic contract checks, not performance measurements."""

import math
import dataclasses
import json
import unittest
from pathlib import Path
from unittest import mock

from tracking.contract import classify
from tests.support import scratch
from utils.config import RolloutConfig
from utils.task_fixtures import negative_sources
from utils.task_policy import diagnostic_rates, gemm_ceiling_issue
from utils.task_suite import evaluate_case


PARAMS = {"m": 4096, "n": 4096, "k": 4096}


def ceiling():
    flops = 2 * 4096 ** 3
    return {"ceiling/available": True, "ceiling/m": 4096, "ceiling/n": 4096,
            "ceiling/k": 4096, "ceiling/dtype": "torch.bfloat16",
            "ceiling/provider": "_BlasBackend.Cublas", "ceiling/warmup_iters": 10,
            "ceiling/rounds": 6, "ceiling/iters_per_round": 10, "ceiling/flops": flops,
            "ceiling/cublas_event_us_median": 200.0, "ceiling/generated_event_us_median": 300.0,
            "ceiling/cublas_fraction": 2 / 3, "ceiling/cublas_tflops": flops / (200 * 1e6),
            "ceiling/generated_tflops": flops / (300 * 1e6)}


class TaskMeasurementTests(unittest.TestCase):
    def test_valid_synthetic_ceiling_and_si_units(self):
        record = ceiling()
        self.assertIsNone(gemm_ceiling_issue(record, PARAMS, "bfloat16"))
        self.assertAlmostEqual(record["ceiling/cublas_tflops"], 687.19476736)

    def test_shape_dtype_provider_and_counts_are_attested(self):
        for key, value, reason in (
                ("ceiling/m", 512, "shape_mismatch"),
                ("ceiling/n", True, "shape_mismatch"),
                ("ceiling/dtype", "torch.float16", "dtype_mismatch"),
                ("ceiling/provider", "CUTLASS profiler", "provider_not_cublas"),
                ("ceiling/rounds", 1, "sampling_mismatch"),
                ("ceiling/flops", 4096 ** 3, "flop_count_mismatch")):
            with self.subTest(key=key):
                record = {**ceiling(), key: value}
                self.assertEqual(gemm_ceiling_issue(record, PARAMS, "bfloat16"), "ceiling_" + reason)

    def test_nonfinite_zero_negative_and_boolean_latencies_fail_closed(self):
        for value in (None, math.nan, math.inf, 0, -1, True):
            record = {**ceiling(), "ceiling/generated_event_us_median": value}
            self.assertEqual(gemm_ceiling_issue(record, PARAMS, "bfloat16"), "ceiling_invalid_latency")

    def test_no_invented_or_inconsistent_rates(self):
        for key in ("ceiling/cublas_fraction", "ceiling/cublas_tflops", "ceiling/generated_tflops"):
            for value in (None, math.nan, 1e9):
                with self.subTest(key=key, value=value):
                    record = {**ceiling(), key: value}
                    self.assertEqual(gemm_ceiling_issue(record, PARAMS, "bfloat16"),
                                     "ceiling_inconsistent_derived_rate")

    def test_missing_ceiling_is_unavailable(self):
        self.assertEqual(gemm_ceiling_issue({}, PARAMS, "bfloat16"), "ceiling_unavailable")

    def test_optional_task_diagnostics_are_table_only(self):
        keys = [*ceiling(),
                *diagnostic_rates({"flops": 100, "logical_io_bytes": 200}, 10),
                "bench/cold_definition", "bench/generated/cold_event_us",
                "bench/compile/cold_wall_us", "task/performance_accepted",
                "task/provenance_authored", "suite/status", "suite/wall_s"]
        for key in keys:
            with self.subTest(key=key):
                self.assertEqual(classify("result", key), "table")

    def test_flash_hack_uses_real_sdpa_and_stable_alias(self):
        source = negative_sources("flash_attention_fwd")["model_new_hack.py"]
        self.assertIn("_sdpa = F.scaled_dot_product_attention", source)
        self.assertIn("return _sdpa(q, k, v,", source)
        self.assertEqual(source.count('BACKEND = "triton"'), 1)

    def test_ceiling_cannot_overwrite_reward_or_reduce_tensor_core_provenance(self):
        captured = []

        def evaluate(workdir, cfg, turns_used, profile_out):
            captured.append(cfg)
            profile_out.update({"provenance/agent": 1.0, "provenance/library": 0.0,
                                "provenance/torch": 0.0, "provenance/memory": 0.0})
            return {"env/verify_ok": True, "env/t_eager_us": 100.0, "env/t_compile_us": 80.0,
                    "env/t_gen_us": 90.0, "env/provenance": 1.0,
                    "reward/value": 2.0, "reward/base": 2, "reward/reason": "faster_than_eager"}

        diagnostic = {**ceiling(), "ceiling/device_name": "NVIDIA H200",
                      "ceiling/device_capability": "9.0", "reward/value": 999,
                      "reward/base": 999, "suite/status": "untrusted_diagnostic"}
        with scratch() as tmp, mock.patch("utils.task_suite.cuda_unavailable", return_value=None), \
                mock.patch("utils.task_suite.available_backends", return_value={"triton": None}), \
                mock.patch("utils.task_suite.run_task", side_effect=evaluate), \
                mock.patch("utils.task_suite._stage", return_value=(0, json.dumps(diagnostic), "", False)):
            result = evaluate_case(
                "gemm_tensor_core", "m4096_n4096_k4096_bfloat16", "triton", "honest",
                Path(tmp) / "case", cfg=dataclasses.replace(
                    RolloutConfig(), benchmark=True, provenance_threshold=.5))
        self.assertEqual(captured[0].provenance_threshold, 1.0)
        self.assertTrue(captured[0].exact_integer_outputs)
        self.assertEqual(result["reward/value"], 2.0)
        self.assertEqual(result["reward/base"], 2)
        self.assertEqual(result["suite/status"], "passed")
        self.assertTrue(result["task/performance_accepted"])


if __name__ == "__main__":
    unittest.main()
