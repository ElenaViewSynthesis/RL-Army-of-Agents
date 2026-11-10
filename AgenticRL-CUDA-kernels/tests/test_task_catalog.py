import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import ROOT, scratch
from utils.task_catalog import TASKS, active_case, case, get_case, get_task, select_cases, validate_usage, workload
from utils.task_suite import case_environment, evaluate_case, expected_outcome, run_suite, select_tasks


class TaskCatalogTests(unittest.TestCase):
    def test_known_curriculum_and_case_identifiers(self):
        self.assertEqual(len(TASKS), 35)
        self.assertEqual([t.name for t in TASKS.values() if t.stage == 0], ["gemm_div_sum_scale"])
        for task in TASKS.values():
            self.assertTrue(task.cases)
            self.assertEqual(len(task.cases), len({item.id for item in task.cases}))
            self.assertTrue((ROOT / "tasks" / task.name / "model.py").is_file())
            for item in task.cases:
                json.dumps(item.to_dict(), allow_nan=False)
                self.assertNotIn("/", item.id)
        self.assertEqual(len(get_task("gemm_div_sum_scale").cases), 54)
        self.assertEqual(get_case("gemm_tensor_core", "m4096_n4096_k4096_bfloat16").params,
                         {"m": 4096, "n": 4096, "k": 4096})

    def test_gpu_mode_tasks_fail_closed_for_training(self):
        for name in ("masked_mean", "prefix_sum", "int8_gemm", "fp8_gemm"):
            validate_usage(name, "eval")
            with self.assertRaisesRegex(ValueError, "eval-only"):
                validate_usage(name, "train")
        validate_usage("grpo_group_norm", "train")

    def test_case_environment_validates_and_restores(self):
        original = dict(os.environ)
        with case_environment("row_softmax", "b64_h128_float32", "cpu"):
            self.assertEqual(active_case("row_softmax").params, {"b": 64, "h": 128})
            with self.assertRaisesRegex(ValueError, "does not match"):
                active_case("masked_mean")
        self.assertEqual(dict(os.environ), original)
        with self.assertRaises(ValueError):
            get_case("row_softmax", "bad")

    def test_grid_selection(self):
        self.assertEqual(len(select_cases("gemm_tensor_core", "acceptance")), 1)
        self.assertEqual(select_cases("flash_attention_fwd", "acceptance")[0].params["s"], 2048)
        self.assertEqual(select_tasks(["bootstrap", "stage1"]),
                         ["gemm_div_sum_scale", "row_softmax", "masked_mean", "prefix_sum"])
        self.assertEqual(len(select_tasks(["all", "stage1"])), 35)

    def test_synthetic_operation_and_byte_counts(self):
        description = workload("gemm_tensor_core", case("bfloat16", m=2, n=3, k=4))
        self.assertEqual((description["flops"], description["logical_io_bytes"]), (48, 76))
        attention = workload("flash_attention_fwd", get_task("flash_attention_fwd").cases[0])
        self.assertEqual((attention["flops"], attention["logical_io_bytes"]),
                         (1075838976, 8388608))
        dense = workload("moe_dense_gemm", get_task("moe_dense_gemm").cases[0])
        self.assertEqual((dense["flops"], dense["logical_io_bytes"]), (25165824, 3158144))
        self.assertIsNone(workload("masked_mean", get_task("masked_mean").cases[0])["flops"])

    def test_no_gpu_is_unavailable_not_a_pass(self):
        with scratch("task-unavailable-") as tmp:
            result = evaluate_case("row_softmax", "b64_h128_float32", "triton", "honest",
                                   Path(tmp) / "case", device="cpu")
            self.assertEqual(result["suite/status"], "unavailable")
            self.assertNotIn("reward/value", result)
            report = run_suite(["row_softmax"], Path(tmp) / "run", device="cpu")
            self.assertEqual(report["counts"], {"passed": 0, "failed": 0, "unavailable": 6})
            self.assertEqual(report["status"], "partial")

    def test_contract_only_is_never_executed(self):
        with patch("utils.task_suite.run_task", side_effect=AssertionError("must not run")):
            result = evaluate_case("fp8_gemm", "m256_n256_k256_float8_e4m3fn",
                                   "triton", "honest", "/unused")
        self.assertEqual((result["suite/status"], result["suite/reason"]),
                         ("unavailable", "contract_only"))

    def test_fixture_expectations_require_actual_gates(self):
        self.assertFalse(expected_outcome({"env/verify_ok": False, "reward/value": -1,
                                          "reward/reason": "compile"}, "wrong"))
        self.assertTrue(expected_outcome({"env/verify_ok": True, "reward/value": -1,
                                         "reward/reason": "provenance"}, "hack"))

    def test_imports_remain_torch_free(self):
        code = ("import sys; import utils.task_catalog, utils.task_policy, utils.task_suite; "
                "assert 'torch' not in sys.modules")
        process = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch unavailable")
class TaskReferenceTests(unittest.TestCase):
    def model(self, name):
        spec = importlib.util.spec_from_file_location("task_reference_" + name,
                                                    ROOT / "tasks" / name / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_reference_edge_cases(self):
        module = self.model("masked_mean")
        actual = module.Model()(torch.tensor([[2., 4.], [3., 9.]]), torch.tensor([0, 2]))
        torch.testing.assert_close(actual, torch.tensor([0., 6.]), atol=1e-2, rtol=1e-2)
        module = self.model("packed_varlen_reduction")
        actual = module.Model()(torch.tensor([2., 4., 9.]), torch.tensor([0, 0, 2, 3]))
        torch.testing.assert_close(actual, torch.tensor([0., 3., 9.]), atol=1e-2, rtol=1e-2)
        module = self.model("prefix_sum")
        self.assertEqual(module.Model()(torch.tensor([1, 2, 3], dtype=torch.int32)).tolist(), [1, 3, 6])

    def test_grpo_reference_matches_repository_population_math(self):
        from utils.grpo import group_advantages
        values = torch.tensor([[1., 3., 5.], [2., 2., 2.]])
        expected = torch.tensor([group_advantages(row) for row in values.tolist()])
        actual = self.model("grpo_group_norm").Model()(values)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_cispo_bounds_from_preset_and_mask(self):
        module = self.model("cispo_clip_weight")
        self.assertEqual(module.get_init_inputs(), [0.0, 5.0])
        result = module.Model()(torch.tensor([[0., 8., -8.]]),
                                torch.tensor([2.]), torch.tensor([[True, True, False]]))
        torch.testing.assert_close(result, torch.tensor([[2., 10., 0.]]), atol=1e-2, rtol=1e-2)

    def test_sampling_is_deterministic_with_stable_ties(self):
        module = self.model("topk_sampling")
        model = module.Model(2, 1.0)
        result = model(torch.tensor([[1., 1., -9.], [1., 1., -9.]]), torch.tensor([0.1, 0.9]))
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.tolist(), [0, 1])
        self.assertEqual(module.Model(2, .4)(
            torch.tensor([[1., 1., -9.]]), torch.tensor([.9])).tolist(), [0])

    def test_adamw_is_functional(self):
        module = self.model("fused_adamw")
        inputs = [torch.tensor([1., 2.]), torch.tensor([.1, -.2]),
                  torch.tensor([0., .1]), torch.tensor([.1, .2])]
        copies = [value.clone() for value in inputs]
        outputs = module.Model()(*inputs)
        self.assertEqual(len(outputs), 3)
        for actual, expected in zip(inputs, copies):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs))

    def test_dense_moe_mask_excludes_unselected_experts(self):
        model = self.model("moe_dense_gemm").Model()
        x = torch.ones(2, 2, dtype=torch.bfloat16)
        gate = torch.ones(2, 2, 2, dtype=torch.bfloat16)
        up = gate.clone()
        down = gate.clone()
        alpha = torch.tensor([[1., 0.], [0., 0.]])
        result = model(x, gate, up, down, alpha)
        self.assertTrue(torch.equal(result[1], torch.zeros(2)))
        self.assertTrue(torch.all(result[0] > 0))


if __name__ == "__main__":
    unittest.main()
