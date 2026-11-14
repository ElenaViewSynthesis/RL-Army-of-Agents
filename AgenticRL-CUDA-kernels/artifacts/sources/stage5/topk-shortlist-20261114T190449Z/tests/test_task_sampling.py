"""Deterministic correctness probes, never timing or reward evidence."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class SamplingFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"CPU interpreter dependencies unavailable: {exc}")
        if not all(importlib.util.find_spec(name) for name in ("triton", "numpy")):
            raise unittest.SkipTest("CPU interpreter requires Triton and NumPy")
        cls.torch = torch
        cls.path = Path(__file__).resolve().parent.parent / (
            "tasks/topk_sampling/kernels/triton/topk_sampling.py")

    def sample(self, logits, uniforms, top_k, top_p):
        script = """
import importlib.util, json, sys, torch
data = json.loads(sys.stdin.read())
spec = importlib.util.spec_from_file_location("_sampling_fixture_probe", data["path"])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = module.topk_sampling_forward(
    torch.tensor(data["logits"], dtype=torch.float32, device="cpu"),
    torch.tensor(data["uniforms"], dtype=torch.float32, device="cpu"),
    data["top_k"], data["top_p"])
print(json.dumps({"dtype": str(result.dtype), "shape": list(result.shape), "values": result.tolist()}))
"""
        data = {"path": str(self.path), "logits": logits.tolist(), "uniforms": uniforms,
                "top_k": top_k, "top_p": top_p}
        process = subprocess.run(
            [sys.executable, "-c", script], input=json.dumps(data), capture_output=True,
            text=True, timeout=120, env={**os.environ, "TRITON_INTERPRET": "1",
                                        "CUDA_AGENT_DEVICE": "cpu", "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout.strip().splitlines()[-1])
        self.assertEqual(result["dtype"], "torch.int32")
        self.assertEqual(result["shape"], [logits.shape[0]])
        return self.torch.tensor(result["values"], dtype=self.torch.int32)

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
