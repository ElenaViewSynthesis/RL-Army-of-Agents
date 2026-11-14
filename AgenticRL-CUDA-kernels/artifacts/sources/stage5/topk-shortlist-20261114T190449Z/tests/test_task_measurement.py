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

    def test_protected_files_are_rechecked_after_optional_ceiling(self):
        def evaluate(workdir, cfg, turns_used, profile_out):
            profile_out.update({"provenance/agent": 1.0, "provenance/library": 0.0,
                                "provenance/torch": 0.0, "provenance/memory": 0.0})
            return {"env/verify_ok": True, "env/t_eager_us": 100.0, "env/t_compile_us": 80.0,
                    "env/t_gen_us": 90.0, "env/provenance": 1.0,
                    "reward/value": 2.0, "reward/base": 2, "reward/reason": "faster_than_eager"}

        diagnostic = {**ceiling(), "ceiling/device_name": "NVIDIA H200",
                      "ceiling/device_capability": "9.0"}
        for name in ("model.py", "binding.cpp", "binding_registry.h"):
            with self.subTest(name=name), scratch() as tmp:
                def stage(module, args, path, environment, deadline):
                    self.assertEqual(module, "utils.task_ceiling")
                    (Path(path) / name).write_text("synthetic late diagnostic-stage mutation\n")
                    return 0, json.dumps(diagnostic), "", False

                with mock.patch("utils.task_suite.cuda_unavailable", return_value=None), \
                        mock.patch("utils.task_suite.available_backends", return_value={"triton": None}), \
                        mock.patch("utils.task_suite.run_task", side_effect=evaluate), \
                        mock.patch("utils.task_suite._stage", side_effect=stage):
                    folder = Path(tmp) / "case"
                    result = evaluate_case(
                        "gemm_tensor_core", "m4096_n4096_k4096_bfloat16", "triton", "honest",
                        folder, cfg=dataclasses.replace(RolloutConfig(), benchmark=True))
                self.assertEqual(result["suite/status"], "failed")
                self.assertEqual(result["suite/reason"], "protected_task_files_modified")
                self.assertEqual(result["reward/value"], -1)
                payload = json.loads((folder / "suite-case.json").read_text())
                self.assertNotEqual(payload["protected_sha256"][name],
                                    payload["protected_sha256_after"][name])


class CuTileGemmGateTests(unittest.TestCase):
    """Synthetic acceptance protocols; not cuTile performance evidence."""

    def case(self, fraction=None, authored=1., benchmark=True, case='m4096_n4096_k4096_bfloat16'):
        def fake_run(path, cfg, turns_used, profile_out):
            self.assertEqual(cfg.provenance_threshold, 1.0)
            profile_out.update({'provenance/agent':authored,'provenance/library':0.,
                                'provenance/torch':1.-authored,'provenance/memory':0.})
            return {'env/verify_ok':True,'env/t_eager_us':100.,'env/t_compile_us':80.,
                    'env/t_gen_us':90.,'env/provenance':authored,'reward/value':2.,
                    'reward/base':2,'reward/reason':'faster_than_eager'}
        record = {**ceiling(), 'ceiling/device_name':'NVIDIA H200', 'ceiling/device_capability':'9.0'}
        if fraction is not None:
            record['ceiling/generated_event_us_median']=record['ceiling/cublas_event_us_median']/fraction
            record['ceiling/cublas_fraction']=fraction
            record['ceiling/generated_tflops']=record['ceiling/flops']/(record['ceiling/generated_event_us_median']*1e6)
        with scratch('cutile-gemm-gate-') as tmp, \
                mock.patch('utils.task_suite.cuda_unavailable',return_value=None), \
                mock.patch('utils.task_suite.available_backends',return_value={'cutile':None}), \
                mock.patch('utils.task_suite.run_task',side_effect=fake_run), \
                mock.patch('utils.task_suite._stage',return_value=(0,json.dumps(record) if fraction is not None else '{}','',False)):
            return evaluate_case('gemm_tensor_core',case,'cutile','honest',Path(tmp)/'case',
                                 cfg=RolloutConfig(benchmark=benchmark,gpu_lock=False,provenance_threshold=.5))

    def test_same_performance_floor_and_reward_is_not_recomputed(self):
        passed=self.case(.6)
        self.assertEqual(passed['task/minimum_cublas_fraction'],.6)
        self.assertTrue(passed['task/performance_required'])
        self.assertTrue(passed['task/performance_accepted'])
        self.assertEqual(passed['suite/status'],'passed')
        failed=self.case(.59)
        self.assertEqual(failed['suite/status'],'failed')
        self.assertEqual(failed['suite/reason'],'below_60_percent_cublas')
        self.assertEqual((passed['reward/value'],failed['reward/value']),(2.,2.))

    def test_missing_or_disabled_measurement_cannot_accept(self):
        for enabled in (False,True):
            with self.subTest(benchmark=enabled):
                result=self.case(benchmark=enabled)
                self.assertEqual(result['suite/status'],'unavailable')
                self.assertIsNone(result['task/performance_accepted'])

    def test_one_hundred_percent_authored_gate_is_not_lowered(self):
        result=self.case(.9,authored=.999)
        self.assertEqual(result['reward/reason'],'provenance')
        self.assertEqual(result['suite/status'],'failed')
        self.assertIsNone(result['task/performance_accepted'])

    def test_smoke_does_not_claim_flagship_acceptance(self):
        result=self.case(benchmark=False,case='m512_n512_k512_bfloat16')
        self.assertFalse(result['task/performance_required'])
        self.assertEqual(result['suite/status'],'passed')

if __name__ == "__main__":
    unittest.main()
