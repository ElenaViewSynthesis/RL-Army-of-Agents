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


@requires("cutlass.cute")
class CuTeDSLAxpbyTests(AxpbyChecks, unittest.TestCase):
    module = "kernels.cutedsl"


class CuTileTaskSourceTests(unittest.TestCase):
    """Source checks only: no cuTile import, compilation or numerical execution."""

    def test_task_layout_lazy_exports_and_kernel_discovery(self):
        import ast
        from pathlib import Path
        from utils.backend import agent_kernel_names, detect_backend
        from tests.support import scratch
        from utils.task_suite import scaffold
        from tests.cutile_source_support import CUTILE_TASKS, task_trees
        for name in CUTILE_TASKS:
            with self.subTest(task=name):
                trees = task_trees(name)
                init = trees[1][1]
                self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == '__getattr__' for n in init.body))
                self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in init.body))
                with scratch('cutile-source-') as tmp:
                    for variant in ('honest', 'hack', 'wrong'):
                        task = scaffold(name, 'cutile', variant, Path(tmp)/variant)
                        self.assertEqual(detect_backend(task), 'cutile')
                        self.assertTrue(agent_kernel_names(task, 'cutile'))
                # Reference and candidate expose the same explicit constructor defaults.
                reference = ast.parse((trees[0][0].parent/'model.py').read_text())
                cls = next(n for n in trees[0][1].body if isinstance(n, ast.ClassDef))
                ref_cls = next(n for n in reference.body if isinstance(n, ast.ClassDef))
                init_fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
                ref_init = next((n for n in ref_cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
                self.assertEqual(ast.dump(init_fn.args) if init_fn else None, ast.dump(ref_init.args) if ref_init else None)

    def test_launch_signatures_and_read_only_kernel_inputs(self):
        import ast
        from tests.cutile_source_support import CUTILE_TASKS, dotted, task_trees
        for name in CUTILE_TASKS:
            path, tree = task_trees(name)[2]
            kernels = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and any(dotted(d)=='ct.kernel' for d in n.decorator_list)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and dotted(node.func)=='ct.launch':
                    self.assertIsInstance(node.args[2], ast.Name)
                    self.assertIsInstance(node.args[3], ast.Tuple)
                    self.assertEqual(len(node.args[3].elts), len(kernels[node.args[2].id].args.args))
            for fn in kernels.values():
                for node in ast.walk(fn):
                    if isinstance(node, ast.Call) and dotted(node.func) in ('ct.scatter','ct.store'):
                        self.assertIn(node.args[0].id, ('out','product','updated','new_m','new_v','summed'))
            self.assertNotIn('tfloat32', path.read_text())

    def test_no_host_tensor_compute_and_only_pinned_api_exports(self):
        import ast
        from tests.cutile_source_support import (CUTILE_TASKS, CT_EXPORTS_USED,
                                                check_host_source, dotted, task_trees)
        for name in CUTILE_TASKS:
            for path, tree in task_trees(name):
                with self.subTest(path=str(path)):
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            is_kernel = any(dotted(d) == 'ct.kernel' for d in node.decorator_list)
                            if not is_kernel:
                                check_host_source(self, node, {a.arg for a in node.args.args})
                        elif isinstance(node, ast.ClassDef):
                            for method in node.body:
                                if isinstance(method, ast.FunctionDef):
                                    check_host_source(self, method, {a.arg for a in method.args.args if a.arg != 'self'})
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Attribute) and dotted(node).startswith('ct.'):
                            self.assertIn(dotted(node).split('.')[1], CT_EXPORTS_USED)
                    if path.stem == name:
                        source = path.read_text()
                        self.assertIn('torch.cuda.current_stream()', source)
                        self.assertIn('with torch.cuda.device(', source)
                        self.assertNotIn('torch.cuda.synchronize', source)

if __name__ == "__main__":
    unittest.main()
