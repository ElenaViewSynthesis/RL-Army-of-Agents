"""Actual subprocess integration; GPU attribution claims require a real GPU.

CPU correctness tests explicitly expect 'unmeasured'. Protocol and classifier
fixtures live in other test modules and never masquerade as GPU evidence.
"""

import dataclasses
import importlib.util
import os
import shutil
import unittest
from unittest.mock import patch

from tests.support import assert_flat, make_task, scratch
from utils.config import ALGOS, RolloutConfig
from utils.grpo import build_batch
from utils.rollout_env import run_task

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRITON = importlib.util.find_spec("triton") is not None
if HAS_TORCH:
    import torch
    HAS_GPU = torch.cuda.is_available()
else:
    HAS_GPU = False


@unittest.skipUnless(HAS_TORCH and HAS_TRITON, "requires Torch + Triton (CPU interpreter is sufficient)")
class CPUIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {"CUDA_AGENT_DEVICE": "cpu", "TRITON_INTERPRET": "1"})
        env.start()
        self.addCleanup(env.stop)
        # Cold Torch imports on a Windows-mounted workdir can exceed the
        # production default before any kernel runs. Do not weaken that default.
        self.cfg = dataclasses.replace(RolloutConfig.preset("grpo"), timeout_s=600)

    def test_honest_interpreter_correctness_no_fabricated_provenance(self):
        task = make_task(self.tmp.name)
        result = run_task(str(task), self.cfg)
        self.assertTrue(result["env/verify_ok"], result)
        self.assertEqual(result["reward/reason"], "unmeasured")
        self.assertIsNone(result["env/provenance"])
        self.assertIsNone(result["env/t_gen_us"])
        self.assertIn("Compile success", result["log/compile_tail"])
        self.assertFalse((task / "cuda_extension.so").exists())
        assert_flat(self, result)

    def test_torch_hack_passes_verification_but_cpu_cannot_attribute(self):
        result = run_task(str(make_task(self.tmp.name, "hack")), self.cfg)
        self.assertTrue(result["env/verify_ok"], result)
        self.assertEqual(result["reward/reason"], "unmeasured")
        self.assertEqual(result["reward/value"], -1)
        assert_flat(self, result)

    def test_timeout_five_seconds_through_run_task(self):
        cfg = dataclasses.replace(RolloutConfig.preset("grpo"), timeout_s=5)
        result = run_task(str(make_task(self.tmp.name, "sleep")), cfg)
        self.assertEqual(result["reward/reason"], "timeout", result)
        self.assertLess(result["env/wall_s"], 8)
        assert_flat(self, result)

    def test_max_turns_through_run_task(self):
        result = run_task(str(make_task(self.tmp.name)), self.cfg, 25)
        self.assertEqual(result["reward/reason"], "max_turns")
        self.assertEqual(result["reward/value"], -1)
        assert_flat(self, result)

    def test_incorrect_output_and_crash_fail_safely(self):
        for kind, reason in (("wrong", "verification"), ("crash", "compile")):
            result = run_task(str(make_task(self.tmp.name, kind)), self.cfg)
            self.assertFalse(result["env/verify_ok"])
            self.assertEqual(result["reward/reason"], reason, result)
            self.assertEqual(result["reward/value"], -1)
            assert_flat(self, result)


@unittest.skipUnless(HAS_GPU and HAS_TRITON and shutil.which("nvcc"), "requires CUDA GPU, nvcc, Torch + Triton")
class GPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = scratch("gpu-")
        cls.addClassCleanup(cls.tmp.cleanup)
        env = patch.dict(os.environ, {"CUDA_AGENT_DEVICE": "cuda", "TRITON_INTERPRET": "0"})
        env.start()
        cls.addClassCleanup(env.stop)
        cfg = RolloutConfig.preset("grpo")
        cls.triton_task = make_task(cls.tmp.name, "triton")
        cls.honest_triton = run_task(str(cls.triton_task), cfg, 1)
        cls.honest_cuda = run_task(str(make_task(cls.tmp.name, "cuda")), cfg, 1)
        cls.hack = run_task(str(make_task(cls.tmp.name, "hack")), cfg, 1)

    def test_honest_triton(self):
        result = self.honest_triton
        self.assertTrue(result["env/verify_ok"], result)
        self.assertGreaterEqual(result["env/provenance"], .9)
        self.assertGreaterEqual(result["reward/value"], 1)
        assert_flat(self, result)

    def test_honest_cuda(self):
        result = self.honest_cuda
        self.assertTrue(result["env/verify_ok"], result)
        self.assertGreaterEqual(result["env/provenance"], .9)
        self.assertGreaterEqual(result["reward/value"], 1)
        assert_flat(self, result)

    def test_torch_hack_passes_verification_fails_provenance(self):
        result = self.hack
        self.assertTrue(result["env/verify_ok"], result)
        self.assertLess(result["env/provenance"], .9)
        self.assertEqual((result["reward/value"], result["reward/reason"]), (-1, "provenance"))
        assert_flat(self, result)

    def test_every_preset_ranks_the_hack_last(self):
        rows = [dict(r, rollout_id=rid) for r, rid in
                ((self.honest_triton, "triton"), (self.honest_cuda, "cuda"), (self.hack, "hack"))]
        for algo in ALGOS:
            cfg = dataclasses.replace(RolloutConfig.preset(algo), group_size=3)
            pairs, stats = build_batch(["axpby"] * 3, rows, cfg)
            advantages = dict(pairs)
            self.assertLess(advantages["hack"], min(advantages["triton"], advantages["cuda"]))
            self.assertAlmostEqual(sum(advantages.values()), 0)
            assert_flat(self, stats)

    def test_dapo_overlong_base_unchanged(self):
        cfg = RolloutConfig.preset("dapo")
        result = run_task(str(self.triton_task), cfg, 19)
        self.assertTrue(result["env/verify_ok"], result)
        self.assertGreater(result["reward/overlong_penalty"], 0)
        # Use the same measured times to avoid comparing noisy milestone crossings.
        from utils.reward import compute_reward
        no_turns = compute_reward(True, result["env/t_eager_us"], result["env/t_compile_us"],
                                  result["env/t_gen_us"], result["env/provenance"], cfg)
        self.assertEqual(result["reward/base"], no_turns["reward_base"])
        assert_flat(self, result)


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


class DSLIntegrationChecks:
    """Mixin: honest and Torch-op hack fixtures for one DSL backend, end to end."""

    backend = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = scratch("gpu-")
        cls.addClassCleanup(cls.tmp.cleanup)
        env = patch.dict(os.environ, {"CUDA_AGENT_DEVICE": "cuda", "TRITON_INTERPRET": "0"})
        env.start()
        cls.addClassCleanup(env.stop)
        cfg = RolloutConfig.preset("grpo")
        cls.honest = run_task(str(make_task(cls.tmp.name, cls.backend)), cfg, 1)
        cls.hack = run_task(str(make_task(cls.tmp.name, f"hack-{cls.backend}")), cfg, 1)

    def test_honest_passes_correctness_and_provenance(self):
        result = self.honest
        self.assertEqual(result["meta/backend"], self.backend)
        self.assertTrue(result["env/verify_ok"], result)
        self.assertGreaterEqual(result["env/provenance"], .9)
        self.assertGreaterEqual(result["reward/value"], 1)
        assert_flat(self, result)

    def test_torch_hack_passes_verification_fails_provenance(self):
        result = self.hack
        self.assertEqual(result["meta/backend"], self.backend)
        self.assertTrue(result["env/verify_ok"], result)
        self.assertLess(result["env/provenance"], .9)
        self.assertEqual((result["reward/value"], result["reward/reason"]), (-1, "provenance"))
        assert_flat(self, result)

    def test_every_preset_ranks_the_hack_last(self):
        rows = [dict(self.honest, rollout_id="honest"), dict(self.hack, rollout_id="hack")]
        for algo in ALGOS:
            cfg = dataclasses.replace(RolloutConfig.preset(algo), group_size=2)
            pairs, _ = build_batch(["axpby"] * 2, rows, cfg)
            advantages = dict(pairs)
            self.assertLess(advantages["hack"], advantages["honest"])


@unittest.skipUnless(HAS_GPU and _has("tilelang"), "requires CUDA GPU, Torch + TileLang")
class TileLangGPUIntegrationTests(DSLIntegrationChecks, unittest.TestCase):
    backend = "tilelang"


@unittest.skipUnless(HAS_GPU and _has("cuda.tile"), "requires CUDA GPU, Torch + cuTile (cuda-tile)")
class CuTileGPUIntegrationTests(DSLIntegrationChecks, unittest.TestCase):
    backend = "cutile"


@unittest.skipUnless(HAS_GPU and _has("cutlass.cute"), "requires CUDA GPU, Torch + CuTe DSL (nvidia-cutlass-dsl)")
class CuTeDSLGPUIntegrationTests(DSLIntegrationChecks, unittest.TestCase):
    backend = "cutedsl"


@unittest.skipUnless(HAS_GPU, "requires CUDA GPU and Torch")
class GenericNameGPUIntegrationTests(unittest.TestCase):
    def test_launched_dummy_kernel_plus_torch_ops_fails_every_backend(self):
        import json
        from utils.backend import BACKENDS, agent_kernel_names, available_backends
        from tests.support import make_spoof_task
        missing = available_backends()
        with scratch("generic-spoof-") as tmp:
            for backend in BACKENDS:
                with self.subTest(backend=backend):
                    if missing[backend] or (backend == "cuda" and not shutil.which("nvcc")):
                        self.skipTest(missing[backend] or "requires nvcc")
                    task = make_spoof_task(tmp, backend)
                    self.assertEqual(agent_kernel_names(task, backend), [])
                    with patch.dict(os.environ, {"CUDA_AGENT_DEVICE": "cuda", "TRITON_INTERPRET": "0"}):
                        result = run_task(str(task), RolloutConfig.preset("grpo"))
                    print(json.dumps({"case": "generic-spoof-" + backend, **result}), flush=True)
                    self.assertTrue(result["env/verify_ok"], result)
                    self.assertLess(result["env/provenance"], .9, result)
                    self.assertEqual((result["reward/value"], result["reward/reason"]), (-1, "provenance"))


if __name__ == "__main__":
    unittest.main()
