"""README rendering reads captured measurements, never reruns benchmarks."""

import unittest

from scripts.task_readme import render


class TaskReadmeTests(unittest.TestCase):
    def record(self, **updates):
        record = {"suite/task": "gemm_tensor_core", "suite/fixture": "honest",
                  "suite/status": "passed", "suite/case": "synthetic-case",
                  "suite/backend": "triton", "bench/available": True,
                  "bench/eager/event_us_median": 12., "bench/compile/event_us_median": 10.,
                  "bench/generated/event_us_median": 8.}
        return {**record, **updates}

    def test_unmeasured_is_not_zero_filled(self):
        text = render("gemm_tensor_core")
        self.assertIn("Unavailable: no measured GPU baseline", text)
        self.assertNotIn("Generated warm us", text)

    def test_measured_rows_need_captured_identity(self):
        for evidence in (None, {"run_id": "synthetic"}, {"command": "synthetic"}):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                render("gemm_tensor_core", [self.record()], evidence)

    def test_nonfinite_and_boolean_times_are_not_presented(self):
        for value in (True, False, 0., -1., float("inf"), float("nan"), None):
            with self.subTest(value=value):
                text = render("gemm_tensor_core", [self.record(
                    **{"bench/generated/event_us_median": value})])
                self.assertIn("Unavailable: no measured GPU baseline", text)

    def test_values_and_provider_identity_are_not_recomputed(self):
        record = self.record(**{
            "ceiling/available": True, "ceiling/provider": "_BlasBackend.Cublas",
            "ceiling/cublas_event_us_median": 20., "ceiling/generated_event_us_median": 16.,
            "ceiling/cublas_tflops": 40., "ceiling/generated_tflops": 50.,
            "ceiling/cublas_fraction": 1.25})
        evidence = {"run_id": "synthetic-display-only", "command": "SYNTHETIC, NOT A GPU RUN"}
        text = render("gemm_tensor_core", [record], evidence, ["Synthetic renderer test only"])
        self.assertIn("12.000000 | 10.000000 | 8.000000", text)
        self.assertIn("unavailable | unavailable | unavailable", text)
        self.assertIn("20.000000 | 16.000000 | 40.000000 | 50.000000 | 125.00%", text)
        self.assertIn("cuBLAS GEMM-only with fp32 output", text)
        self.assertIn("Synthetic renderer test only", text)


if __name__ == "__main__":
    unittest.main()
