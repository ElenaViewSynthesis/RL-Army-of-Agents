import importlib.util
import json
import os
import subprocess
import sys
import types
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
            self.assertEqual(report["not_applicable"], 0)
            self.assertEqual(report["status"], "partial")

    def test_not_applicable_controls_do_not_make_a_run_partial(self):
        # SYNTHETIC rows: one pass plus one control with nothing to test is a pass;
        # any other unavailable row still makes the run partial.
        passed = {"suite/status": "passed", "suite/reason": None}
        skip = {"suite/status": "unavailable", "suite/reason": "control_not_applicable_no_skipped_tokens"}
        other = {"suite/status": "unavailable", "suite/reason": "cuda_unavailable"}
        for rows, status, count in (([passed, skip], "pass", 1), ([passed, skip, other], "partial", 1)):
            with self.subTest(status=status), scratch("task-na-") as tmp,                     patch("utils.task_suite.evaluate_case", side_effect=[dict(r) for r in rows]),                     patch("utils.task_suite.select_cases", return_value=[types.SimpleNamespace(id="c")] ):
                report = run_suite(["row_softmax"], Path(tmp) / "run", device="cpu",
                                   backends=("triton",), variants=("honest",) * len(rows))
                self.assertEqual((report["status"], report["not_applicable"]), (status, count))

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


class CuTileOptInCatalogTests(unittest.TestCase):
    def test_existing_cases_and_default_backends_are_unchanged(self):
        import hashlib
        from tests.cutile_source_support import CUTILE_TASKS
        cases = {n: [c.to_dict() for c in t.cases] for n, t in TASKS.items()}
        self.assertEqual(hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                         'b00890e4555a494954ece8fd7af57d8de0e411aa773e9f696c2fec69a452b6c0')
        defaults = {n: list(t.backends) for n, t in TASKS.items()}
        self.assertEqual(hashlib.sha256(json.dumps(defaults, sort_keys=True).encode()).hexdigest(),
                         '0bdccd9f5a4f75c58b411f98acebfe14f4dfa5f5a9f98e5ac4fda82b43bb282c')
        for name in CUTILE_TASKS:
            self.assertIn('cutile', get_task(name).stretch_backends)
            self.assertNotIn('cutile', get_task(name).backends)

    def test_readme_static_contract_matches_renderer(self):
        from scripts.task_readme import render
        from tests.cutile_source_support import CUTILE_TASKS
        marker = 'Unavailable: no measured GPU baseline has been recorded here.'
        for name in CUTILE_TASKS:
            prefix = render(name).split(marker)[0]
            self.assertTrue((ROOT/'tasks'/name/'README.md').read_text().startswith(prefix), name)
            self.assertIn('cuTile timing, speedup and provenance are **not measured**', prefix)

    def test_default_rows_do_not_add_stretch_backends(self):
        import contextlib
        import io
        from tests.cutile_source_support import CUTILE_TASKS
        with scratch('cutile-default-') as tmp, contextlib.redirect_stdout(io.StringIO()):
            report = run_suite(CUTILE_TASKS, Path(tmp)/'suite', device='cpu')
        expected = sum(3*len(get_task(n).backends) for n in CUTILE_TASKS)
        self.assertEqual(len(report['rows']), expected)
        self.assertFalse(any(r['suite/backend']=='cutile' for r in report['rows']))

    def test_explicit_cutile_selection_reaches_all_three_fixture_gates(self):
        # Synthetic subprocess protocol results; no compilation or GPU measurement.
        from tests.cutile_source_support import CUTILE_TASKS
        from utils.backend import detect_backend
        def fake_run(path, cfg, turns_used, profile_out):
            self.assertEqual(detect_backend(path), 'cutile')
            self.assertEqual(cfg.backend, 'cutile')
            variant = Path(path).name
            if variant == 'wrong':
                return {'env/verify_ok': False, 'reward/value': -1, 'reward/base': -1,
                        'reward/reason': 'verification'}
            authored = 1. if variant == 'honest' else 0.
            profile_out.update({'provenance/agent': authored, 'provenance/library': 0.,
                                'provenance/torch': 1.-authored, 'provenance/memory': 0.})
            return {'env/verify_ok': True, 'env/t_eager_us': 100., 'env/t_compile_us': 80.,
                    'env/t_gen_us': 50., 'env/provenance': authored,
                    'reward/value': 3 if authored else -1, 'reward/base': 3 if authored else -1,
                    'reward/reason': 'faster_than_both' if authored else 'provenance'}
        with scratch('cutile-explicit-') as tmp, \
                patch('utils.task_suite.cuda_unavailable', return_value=None), \
                patch('utils.task_suite.available_backends', return_value={'cutile':None}), \
                patch('utils.task_suite.run_task', side_effect=fake_run):
            for name in CUTILE_TASKS:
                for variant in ('honest', 'hack', 'wrong'):
                    result = evaluate_case(name, get_task(name).cases[0].id, 'cutile', variant,
                                           Path(tmp)/name/variant)
                    self.assertEqual(result['suite/status'], 'passed', result)

if __name__ == "__main__":
    unittest.main()
