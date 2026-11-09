"""Deterministic correctness probes, never timing or reward evidence."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


class SamplingFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import triton
            import numpy
        except ImportError as exc:
            raise unittest.SkipTest(f"CPU interpreter dependencies unavailable: {exc}")
        cls.torch = torch
        path = Path(__file__).resolve().parent.parent / (
            "tasks/topk_sampling/kernels/triton/topk_sampling.py")
        with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
            spec = importlib.util.spec_from_file_location("_sampling_fixture_probe", path)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def sample(self, logits, uniforms, top_k, top_p):
        with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
            return self.module.topk_sampling_forward(
                logits, self.torch.tensor(uniforms, dtype=self.torch.float32), top_k, top_p)

    def test_stable_equal_logits_and_exact_cdf_boundary(self):
        logits = self.torch.tensor([[2., 2., 1., -1.]] * 3)
        actual = self.sample(logits, [0., .5, .999], 2, 1.)
        self.torch.testing.assert_close(actual, self.torch.tensor([0, 1, 1], dtype=self.torch.int32))

    def test_signed_zeros_are_ties_not_distinct_float_bit_keys(self):
        actual = self.sample(self.torch.tensor([[-0., 0., -2., -3.]]), [.999], 1, 1.)
        self.torch.testing.assert_close(actual, self.torch.tensor([0], dtype=self.torch.int32))

    def test_top_p_retains_crossing_token_then_renormalizes(self):
        import math

        logits = self.torch.tensor([[math.log(.6), math.log(.3), math.log(.1)]] * 4)
        actual = self.sample(logits, [0., .66, .67, .999], 3, .7)
        self.torch.testing.assert_close(actual, self.torch.tensor([0, 0, 1, 1], dtype=self.torch.int32))

    def test_top_p_excludes_token_with_prefix_equal_to_threshold(self):
        logits = self.torch.tensor([[0., 0., -3., -4.]] * 2)
        actual = self.sample(logits, [.1, .999], 2, .5)
        self.torch.testing.assert_close(actual, self.torch.tensor([0, 0], dtype=self.torch.int32))

    def test_cross_chunk_tie_and_partial_final_chunk(self):
        logits = self.torch.full((2, 2051), -1000., dtype=self.torch.float32)
        logits[:, 1023] = 0.
        logits[:, 2048] = 0.
        actual = self.sample(logits, [.25, .75], 2, 1.)
        self.torch.testing.assert_close(
            actual, self.torch.tensor([1023, 2048], dtype=self.torch.int32))


if __name__ == "__main__":
    unittest.main()
