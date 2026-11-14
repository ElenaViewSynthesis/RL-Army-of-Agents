"""Stage 7 (TileGym Tier 1) contracts: catalog, selection, references, fixtures and READMEs.

No GPU: references run on CPU tensors; Triton, CUDA and cuTile fixtures are checked as
source and layout only here and on the H200 by the task suite.
"""

import importlib.util
import os
import unittest
from pathlib import Path

from tests.support import ROOT
from utils.task_catalog import TASKS, case, get_task, select_cases, workload

STAGE7 = ("relu_squared", "silu_and_mul", "geglu_tanh", "rotary_embedding")
HAS_TORCH = importlib.util.find_spec("torch") is not None


def load_model(task):
    spec = importlib.util.spec_from_file_location(f"stage7_{task}_model", ROOT / "tasks" / task / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage7CatalogTests(unittest.TestCase):
    def test_tasks_backends_and_sources(self):
        self.assertEqual([n for n, t in TASKS.items() if t.stage == 7], list(STAGE7))
        for name in STAGE7:
            task = get_task(name)
            with self.subTest(task=name):
                self.assertEqual(task.backends, ("triton", "cuda"))
                self.assertEqual(task.stretch_backends, ("cutile",))
                self.assertIn("tilegym", task.sources)
                self.assertFalse(task.eval_only or task.contract_only)
                self.assertEqual(len(task.cases), 12)
        self.assertIn("liger", get_task("relu_squared").sources)

    def test_stage_selection_and_largest_acceptance(self):
        from utils.task_suite import select_tasks
        self.assertEqual(select_tasks(["stage7"]), list(STAGE7))
        expected = {"relu_squared": ("b64_h1024_float32", "b4096_h14336_bfloat16"),
                    "silu_and_mul": ("b64_h1024_float32", "b4096_h14336_bfloat16"),
                    "geglu_tanh": ("b64_h1024_float32", "b4096_h16384_bfloat16"),
                    "rotary_embedding": ("b1_hq32_hkv8_s512_d64_float32",
                                         "b8_hq32_hkv8_s8192_d128_bfloat16")}
        for name, (smoke, acceptance) in expected.items():
            with self.subTest(task=name):
                self.assertEqual(select_cases(name, "smoke")[0].id, smoke)
                self.assertEqual(select_cases(name, "acceptance")[0].id, acceptance)
                self.assertEqual(get_task(name).cases[-1].id, acceptance)

    def test_earlier_stages_keep_their_acceptance_choice(self):
        # Stage 7 widened select_cases to (5, 6, 7); stages 0-4 still take the first case.
        self.assertEqual(select_cases("rmsnorm", "acceptance"), get_task("rmsnorm").cases[:1])
        self.assertEqual(select_cases("log_softmax_gather", "acceptance"),
                         get_task("log_softmax_gather").cases[:1])

    def test_workload_accounting(self):
        self.assertEqual(workload("relu_squared", case("bfloat16", b=2, h=3))["logical_io_bytes"], 24)
        self.assertEqual(workload("silu_and_mul", case("float32", b=2, h=3))["logical_io_bytes"], 72)
        rope = workload("rotary_embedding", case("bfloat16", b=1, hq=2, hkv=1, s=4, d=8))
        self.assertEqual(rope["flops"], 3 * 3 * 4 * 8)
        self.assertEqual(rope["logical_io_bytes"], 2 * 96 * 2 + 2 * 4 * 4 * 4)


class Stage7FixtureTests(unittest.TestCase):
    def test_layout_and_generated_negative_controls(self):
        from utils.task_fixtures import negative_sources
        for name in STAGE7:
            base = ROOT / "tasks" / name
            with self.subTest(task=name):
                for rel in ("model.py", "model_new.py", "model_new_triton.py", "model_new_cuda.py",
                            "model_new_cutile.py", f"kernels/{name}.cu", f"kernels/{name}_binding.cpp",
                            f"kernels/triton/{name}.py", f"kernels/cuda/{name}.py",
                            f"kernels/cutile/{name}.py", "README.md"):
                    self.assertTrue((base / rel).is_file(), rel)
                for filename, source in negative_sources(name).items():
                    self.assertEqual((base / filename).read_text(), source)

    def test_readmes_match_renderer(self):
        from scripts.task_readme import render
        for name in STAGE7:
            with self.subTest(task=name):
                self.assertEqual((ROOT / "tasks" / name / "README.md").read_text(), render(name))

    def test_kernels_use_fp32_and_fresh_outputs(self):
        for name in STAGE7:
            for backend in ("triton", "cutile"):
                text = (ROOT / "tasks" / name / "kernels" / backend / f"{name}.py").read_text()
                with self.subTest(task=name, backend=backend):
                    self.assertIn("float32", text)
                    self.assertNotIn("exhaustive_search", text)
                    self.assertNotIn("get_device_capability", text)
                    self.assertNotIn("ByTarget", text)
                    self.assertTrue("empty_like" in text or "torch.empty(" in text)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class Stage7ReferenceTests(unittest.TestCase):
    """CPU checks of the trusted references against independent formulas (not GPU evidence)."""

    def setUp(self):
        import torch
        torch.manual_seed(0)

    def test_activations_match_independent_formulas(self):
        import torch
        import torch.nn.functional as F
        x = torch.randn(8, 2 * 300)
        h = 300
        self.assertTrue(torch.allclose(load_model("relu_squared").Model()(x), F.relu(x) ** 2))
        self.assertTrue(torch.allclose(load_model("silu_and_mul").Model()(x), F.silu(x[:, :h]) * x[:, h:], atol=1e-6))
        geglu = load_model("geglu_tanh").Model()(x)
        self.assertTrue(torch.allclose(geglu, x[:, :h] * F.gelu(x[:, h:], approximate="tanh"), atol=1e-6))
        # The split direction is part of the contract: swapping the halves must differ.
        self.assertFalse(torch.allclose(geglu, F.gelu(x[:, :h], approximate="tanh") * x[:, h:], atol=1e-2))

    def test_bf16_output_dtype_and_inputs_unchanged(self):
        import torch
        x = torch.randn(4, 2048, dtype=torch.bfloat16)
        before = x.clone()
        for name in ("relu_squared", "silu_and_mul", "geglu_tanh"):
            out = load_model(name).Model()(x)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(x, before))

    def test_rotary_matches_complex_rotation_and_returns_new_tensors(self):
        import torch
        b, s, d = 2, 32, 16
        q, k = torch.randn(b, 4, s, d), torch.randn(b, 2, s, d)
        half = d // 2
        inv = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) * 2 / d))
        angles = torch.arange(s, dtype=torch.float32)[:, None] * inv[None, :]
        cos, sin = torch.cos(angles), torch.sin(angles)
        q0, k0 = q.clone(), k.clone()
        q_rot, k_rot = load_model("rotary_embedding").Model()(q, k, cos, sin)
        for x, out in ((q, q_rot), (k, k_rot)):
            z = torch.complex(x[..., :half], x[..., half:]) * torch.polar(torch.ones_like(angles), angles)
            self.assertTrue(torch.allclose(out, torch.cat((z.real, z.imag), dim=-1), atol=1e-5))
        self.assertTrue(torch.equal(q, q0) and torch.equal(k, k0))
        self.assertIsNot(q_rot, q)

    def test_generated_inputs_have_contract_shapes(self):
        import torch
        from utils.task_inputs import make_inputs
        from utils.task_suite import case_environment
        for name in STAGE7:
            item = select_cases(name, "smoke")[0]
            with self.subTest(task=name), case_environment(name, item.id, "cpu"):
                inputs = make_inputs(name)
                p = item.params
                if name == "rotary_embedding":
                    self.assertEqual(tuple(inputs[0].shape), (p["b"], p["hq"], p["s"], p["d"]))
                    self.assertEqual(tuple(inputs[1].shape), (p["b"], p["hkv"], p["s"], p["d"]))
                    self.assertEqual(tuple(inputs[2].shape), (p["s"], p["d"] // 2))
                    self.assertEqual(inputs[2].dtype, torch.float32)
                else:
                    width = p["h"] if name == "relu_squared" else 2 * p["h"]
                    self.assertEqual(tuple(inputs[0].shape), (p["b"], width))


if __name__ == "__main__":
    unittest.main()
