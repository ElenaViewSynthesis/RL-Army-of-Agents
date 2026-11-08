import importlib.util
import os
import subprocess
import sys
import unittest

from tests.support import ROOT


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("triton"),
                     "requires Torch + Triton for CPU interpretation")
class TritonKernelTests(unittest.TestCase):
    def test_interpreter_shapes_masking_dtypes_and_validation(self):
        code = '''
import torch
from kernels.triton import axpby_forward
torch.manual_seed(17)
for dtype in (torch.float16, torch.float32, torch.float64):
    for shape in ((), (0,), (1,), (127,), (129,), (3, 257), (1, 128)):
        a, b = torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype)
        for alpha in (-2., 0., .25, 2.):
            before_a, before_b = a.clone(), b.clone()
            actual = axpby_forward(a, b, alpha)
            torch.testing.assert_close(actual, alpha * a + b, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(a, before_a)
            torch.testing.assert_close(b, before_b)
            assert actual.shape == a.shape and actual.dtype == a.dtype
            if a.numel():
                assert actual.data_ptr() != a.data_ptr()
for a, b in ((torch.randn(3), torch.randn(4)),
             (torch.randn(8)[::2], torch.randn(4)),
             (torch.ones(4, dtype=torch.int32), torch.ones(4, dtype=torch.int32))):
    try:
        axpby_forward(a, b, 2.)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid inputs were accepted")
print("CPU interpreter: scalar/empty/tail/matrix shapes, 3 dtypes, 4 alphas passed")
'''
        env = {**os.environ, "CUDA_AGENT_DEVICE": "cpu", "TRITON_INTERPRET": "1"}
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True, timeout=600)


if __name__ == "__main__":
    unittest.main()
