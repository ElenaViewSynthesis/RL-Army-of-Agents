import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.support import ROOT, assert_flat, make_task, scratch
from utils.backend import agent_kernel_names, detect_backend
from utils.provenance import classify_kernel, summarize_events


class BackendTests(unittest.TestCase):
    def test_protected_upstream_files_are_byte_identical(self):
        manifest = json.loads((ROOT / "artifacts/upstream.json").read_text())
        for name in ("model.py", "binding.cpp", "binding_registry.h", "model_new.py",
                     "kernels/axpby.cu", "kernels/axpby_binding.cpp"):
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), manifest["sha256"][name])

    def test_reward_hack_torchop_fixture_backend_selection(self):
        from utils.backend import BACKENDS
        for backend in BACKENDS:
            with self.subTest(backend=backend), scratch() as tmp:
                task = make_task(tmp, f"reward-hack-torchop-{backend}")
                self.assertEqual(detect_backend(task), backend)
        with scratch() as tmp:
            task = make_task(tmp, "reward-hack-torchop")
            self.assertEqual(detect_backend(task), "triton")

    def test_stock_source_discovery(self):
        self.assertEqual(detect_backend(ROOT), "cuda")
        self.assertEqual(agent_kernel_names(ROOT), ["axpby_kernel"])
        self.assertEqual(agent_kernel_names(ROOT, "triton"), ["axpby_kernel"])

    def test_imports_overrides_and_no_execution(self):
        with scratch() as tmp:
            path = Path(tmp)
            for source, expected in (
                ("import cuda_extension as c", "cuda"),
                ("from cuda_extension import forward", "cuda"),
                ("from kernels.triton.axpby import axpby_forward", "triton"),
                ("from kernels import triton", "triton"),
                ("import kernels.triton.axpby", "triton"),
                ('BACKEND: str = "triton"\nimport cuda_extension', "triton"),
                ('BACKEND = "cuda"\nfrom kernels.triton import forward', "cuda"),
                ('raise RuntimeError("must not execute")', "cuda"),
            ):
                with self.subTest(source=source):
                    (path / "model_new.py").write_text(source)
                    self.assertEqual(detect_backend(path), expected)
            for source in ('BACKEND = "invalid"', 'BACKEND = compute()',
                           'import cuda_extension\nimport kernels.triton'):
                (path / "model_new.py").write_text(source)
                with self.assertRaises(ValueError):
                    detect_backend(path)

    def test_decorators_comments_multiline_cuda(self):
        with scratch() as tmp:
            path = make_task(tmp)
            (path / "kernels/triton/aliased.py").write_text(
                'import triton as t\nfrom triton import jit as j\n'
                '@t.jit\ndef first(): pass\n@j()\ndef second(): pass\ndef not_jit(): pass\n')
            self.assertEqual(agent_kernel_names(path), ["axpby_kernel", "first", "second"])
            (path / "kernels/extra.cu").write_text(
                '// __global__ void fake() {}\n/* __global__ void fake2() {} */\n'
                'template<int N>\n__global__ __launch_bounds__(256)\nvoid extra(float* a) {}\n')
            self.assertEqual(agent_kernel_names(path, "cuda"), ["axpby_kernel", "extra"])


class DSLBackendTests(unittest.TestCase):
    """Source discovery for the DSL backends; needs none of the DSL packages."""

    def check_backend(self, backend, stock_names, alias_source, alias_names):
        with scratch() as tmp:
            path = make_task(tmp, backend)
            self.assertEqual(detect_backend(path), backend)
            self.assertEqual(agent_kernel_names(path), stock_names)
            for source in (f"from kernels.{backend} import axpby_forward",
                           f"import kernels.{backend}.axpby",
                           f'BACKEND = "{backend}"\nimport cuda_extension'):
                with self.subTest(source=source):
                    (path / "model_new.py").write_text(source)
                    self.assertEqual(detect_backend(path), backend)
            (path / "model_new.py").write_text(f"import cuda_extension\nimport kernels.{backend}")
            with self.assertRaises(ValueError):
                detect_backend(path)
            (path / f"kernels/{backend}/aliased.py").write_text(alias_source)
            self.assertEqual(agent_kernel_names(path, backend), sorted(stock_names + alias_names))

    def check_no_library_credit(self, backend, agent_name, symbol):
        self.assertEqual(classify_kernel(symbol, [agent_name], backend), "agent")
        for name in ("cublasLtMatmul", "cutlass_gemm", "cudnn_conv", "aten::mul", "unknown"):
            self.assertEqual(classify_kernel(name, [agent_name], backend), "torch")

    def test_tilelang_detection_and_discovery(self):
        self.check_backend("tilelang", ["axpby_tilelang"], (
            "import tilelang\nimport tilelang.language as T\nfrom tilelang import language as L\n"
            "import tilelang.cuda.language as CT\n"
            "@T.prim_func\ndef first(): pass\n@L.prim_func\ndef second(): pass\n"
            "@CT.prim_func\ndef third(): pass\n@tilelang.jit(out_idx=[-1])\ndef fourth(): pass\n"
            "@T.macro\ndef not_a_kernel(): pass\ndef plain(): pass\n"),
            ["first", "fourth", "second", "third"])

    def test_tilelang_library_time_not_credited(self):
        self.check_no_library_credit("tilelang", "axpby_tilelang", "axpby_tilelang_kernel")

    def test_cutile_detection_and_discovery(self):
        self.check_backend("cutile", ["axpby_cutile_kernel"], (
            "import cuda.tile as ct\nfrom cuda import tile as tl\nfrom cuda.tile import kernel\n"
            "import cuda.tile\n"
            "@ct.kernel\ndef first(a): pass\n@tl.kernel(occupancy=2)\ndef second(a): pass\n"
            "@kernel\ndef third(a): pass\n@cuda.tile.kernel\ndef fourth(a): pass\n"
            "def host_wrapper(a): pass\n"),
            ["first", "fourth", "second", "third"])

    def test_cutile_library_time_not_credited(self):
        self.check_no_library_credit("cutile", "axpby_cutile_kernel", "axpby_cutile_kernel_Kt1_A1f32")

    def test_cutedsl_detection_and_discovery(self):
        self.check_backend("cutedsl", ["axpby_cute_kernel"], (
            "import cutlass\nimport cutlass.cute as cute\nfrom cutlass import cute as cu\n"
            "from cutlass.cute import kernel\n"
            "@cute.kernel\ndef first(a): pass\n@cu.kernel\ndef second(a): pass\n"
            "@kernel\ndef third(a): pass\n@cutlass.cute.kernel\ndef fourth(a): pass\n"
            "@cute.jit\ndef host_launcher(a): pass\n"),
            ["first", "fourth", "second", "third"])

    def test_cutedsl_cutlass_named_symbols(self):
        # Default CuTe DSL symbols embed "cutlass": credited only via the
        # authored name, never as library time on this backend.
        symbol = "kernel_cutlass_axpby_cute_kernel_tensorptrf32gmem_0"
        self.assertEqual(classify_kernel(symbol, ["axpby_cute_kernel"], "cutedsl"), "agent")
        self.assertEqual(classify_kernel(symbol, [], "cutedsl"), "torch")
        self.assertEqual(classify_kernel(symbol, [], "cuda"), "library")
        self.check_no_library_credit("cutedsl", "axpby_cute_kernel", symbol)


class ProvenanceTests(unittest.TestCase):
    def test_cpu_profiling_skips_without_importing_torch(self):
        proc = subprocess.run([sys.executable, "-S", "-m", "utils.profiling", "--json"], cwd=ROOT,
                              env={**os.environ, "CUDA_AGENT_DEVICE": "cpu"},
                              capture_output=True, text=True, check=True)
        result = json.loads(proc.stdout.splitlines()[-1])
        self.assertFalse(result["timing_measured"])
        self.assertIsNone(result["provenance"])
        self.assertIsNone(result["t_gen_us"])
        assert_flat(self, result)

    def test_legacy_stdout_format_without_importing_torch(self):
        tree = ast.parse((ROOT / "utils/profiling.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "print_results")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "profiling.print_results", "exec"), namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            namespace["print_results"](10, 8, 5)
        self.assertEqual(output.getvalue(), "Torch Baseline: 10.000us, Torch Compile: 8.000us, CUDA Extension: 5.000us\n")

    def test_classes_and_library_backend_policy(self):
        self.assertEqual(classify_kernel("void axpby_kernel<256>(float*)", ["axpby_kernel"], "cuda"), "agent")
        self.assertEqual(classify_kernel("axpby_kernel_0d1d2d", ["axpby_kernel"], "triton"), "agent")
        for backend in ("cuda", "triton"):
            for name in ("at::native::vectorized_elementwise_kernel", "aten::mul", "triton_poi_fused_mul_add_0", "unknown"):
                self.assertEqual(classify_kernel(name, ["axpby_kernel"], backend), "torch")
            for name in ("cublasLtMatmul", "cutlass_gemm", "cudnn_conv"):
                self.assertEqual(classify_kernel(name, [], backend), "library" if backend == "cuda" else "torch")
            for name in ("Memcpy HtoD", "[CUDA memset]", "Memset axpby_kernel"):
                self.assertEqual(classify_kernel(name, ["axpby_kernel"], backend), "memory")

    def test_memory_excluded_from_gate_and_per_kernel_averaging(self):
        events = [("axpby_kernel", 90), ("cublas_gemm", 5), ("aten::add", 5), ("Memcpy DtoD", 900)]
        cuda = summarize_events([("void axpby_kernel<256>(float*)", 90), *events[1:]],
                                ["axpby_kernel"], "cuda", iters=10)
        triton = summarize_events(events, ["axpby_kernel"], "triton", iters=10)
        self.assertAlmostEqual(cuda["provenance"], .95)
        self.assertAlmostEqual(triton["provenance"], .9)
        self.assertEqual(cuda["kernel_us/void axpby_kernel<256>(float*)"], 9)
        self.assertEqual(cuda["t_gen_us"], 100)
        self.assertAlmostEqual(sum(cuda["provenance/" + k] for k in ("agent", "library", "torch", "memory")), 1)
        assert_flat(self, cuda)

    def test_hack_unknown_and_empty_fail_closed(self):
        hack = summarize_events([("at::native::mul", 1), ("at::native::add", 1)], ["axpby_kernel"], "triton")
        self.assertEqual(hack["provenance"], 0)
        self.assertEqual(summarize_events([], [], "triton")["provenance"], 0)
        self.assertEqual(summarize_events([("Memset", 100)], [], "cuda")["provenance"], 0)
        with self.assertRaises(ValueError):
            summarize_events([("bad", float("nan"))], [], "cuda")


if __name__ == "__main__":
    unittest.main()
