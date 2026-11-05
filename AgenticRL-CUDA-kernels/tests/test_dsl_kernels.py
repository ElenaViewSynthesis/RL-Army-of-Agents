"""GPU-only correctness for the DSL backends' example kernels.

None of these DSLs ships a CPU interpreter in its release packages, so every
test here skips without the package and a CUDA device. Results from these
tests are the only evidence that the example kernels work on hardware.
"""

import importlib
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    HAS_GPU = torch.cuda.is_available()
else:
    HAS_GPU = False

SHAPES = ((), (0,), (1,), (127,), (129,), (3, 257), (1, 128), (4096,))
DTYPES = ("float16", "bfloat16", "float32", "float64")
ALPHAS = (-2.0, 0.25, 2.0)


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def requires(module):
    return unittest.skipUnless(HAS_GPU and _has(module), f"requires a CUDA GPU, Torch and {module}")


class AxpbyChecks:
    """Mixin: set `module` to the kernel package exporting axpby_forward."""

    module = None

    def forward(self):
        return importlib.import_module(self.module).axpby_forward

    def test_shapes_tails_dtypes_and_inputs_unchanged(self):
        axpby_forward = self.forward()
        torch.manual_seed(17)
        for dtype in DTYPES:
            for shape in SHAPES:
                a = torch.randn(shape, dtype=getattr(torch, dtype), device="cuda")
                b = torch.randn(shape, dtype=getattr(torch, dtype), device="cuda")
                for alpha in ALPHAS:
                    with self.subTest(dtype=dtype, shape=shape, alpha=alpha):
                        before_a, before_b = a.clone(), b.clone()
                        actual = axpby_forward(a, b, alpha)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, alpha * a + b, atol=1e-2, rtol=1e-2)
                        torch.testing.assert_close(a, before_a)
                        torch.testing.assert_close(b, before_b)
                        self.assertEqual((actual.shape, actual.dtype, actual.device),
                                         (a.shape, a.dtype, a.device))
                        if a.numel():
                            self.assertNotEqual(actual.data_ptr(), a.data_ptr())

    def test_invalid_inputs_rejected(self):
        axpby_forward = self.forward()
        cuda = {"device": "cuda"}
        for a, b in ((torch.randn(3, **cuda), torch.randn(4, **cuda)),
                     (torch.randn(8, **cuda)[::2], torch.randn(4, **cuda)),
                     (torch.ones(4, dtype=torch.int32, **cuda), torch.ones(4, dtype=torch.int32, **cuda)),
                     (torch.randn(4), torch.randn(4))):
            with self.subTest(shape=tuple(a.shape), dtype=a.dtype, device=a.device.type):
                with self.assertRaises(ValueError):
                    axpby_forward(a, b, 2.0)

    def test_graph_capture_replays_on_current_stream(self):
        # A kernel launched on any stream other than the capturing (current)
        # stream either breaks capture or is not replayed, leaving `out` stale.
        axpby_forward = self.forward()
        a = torch.randn(1000, device="cuda")
        b = torch.randn(1000, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                axpby_forward(a, b, 2.0)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = axpby_forward(a, b, 2.0)
        a.copy_(torch.randn(1000, device="cuda"))
        b.copy_(torch.randn(1000, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, 2.0 * a + b, atol=1e-2, rtol=1e-2)


@requires("tilelang")
class TileLangAxpbyTests(AxpbyChecks, unittest.TestCase):
    module = "kernels.tilelang"


@requires("cuda.tile")
class CuTileAxpbyTests(AxpbyChecks, unittest.TestCase):
    module = "kernels.cutile"


if __name__ == "__main__":
    unittest.main()
