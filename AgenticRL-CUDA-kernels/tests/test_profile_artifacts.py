"""Opt-in diagnostic profiling artifacts (Step 2); GPU checks skip explicitly."""

import dataclasses
import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import ARTIFACT_KEYS, DEFAULT_RESULT_KEYS, ROOT, assert_flat, make_task, scratch
from utils import artifact_report
from utils.config import RolloutConfig
from utils.rollout_env import _stage, run_task

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRITON = importlib.util.find_spec("triton") is not None
if HAS_TORCH:
    import torch
    HAS_GPU = torch.cuda.is_available()
else:
    HAS_GPU = False

# SYNTHETIC profiler events for aggregation tests; never a GPU measurement.
SYNTHETIC_EVENTS = [("axpby_kernel", 2.0), ("axpby_kernel", 4.0), ("aten::add", 1.0),
                    ("Memcpy HtoD (Pageable -> Device)", 3.0), ("unknown_thing", 0.5)]


def canned_stages(provenance=1.0):
    from tests.test_rollout_env import RolloutProtocolTests
    return RolloutProtocolTests.stages(None, provenance)


class KernelTableTests(unittest.TestCase):
    def test_aggregation_classes_and_order_synthetic(self):
        rows = artifact_report.kernel_table(SYNTHETIC_EVENTS, ["axpby_kernel"], "triton", iters=2)
        self.assertEqual([r["name"] for r in rows],
                         ["axpby_kernel", "Memcpy HtoD (Pageable -> Device)", "aten::add",
                          "unknown_thing"])
        top = rows[0]
        self.assertEqual((top["calls"], top["total_us"], top["mean_us"], top["per_iter_us"],
                          top["class"]), (2, 6.0, 3.0, 3.0, "agent"))
        classes = {r["name"]: r["class"] for r in rows}
        self.assertEqual(classes["aten::add"], "torch")
        self.assertEqual(classes["Memcpy HtoD (Pageable -> Device)"], "memory")
        self.assertEqual(classes["unknown_thing"], "torch")  # Unknown fails closed.

    def test_rejects_nonfinite_and_bad_iters(self):
        with self.assertRaises(ValueError):
            artifact_report.kernel_table([("k", float("nan"))], [], "triton", 1)
        with self.assertRaises(ValueError):
            artifact_report.kernel_table([("k", -1.0)], [], "triton", 1)
        with self.assertRaises(ValueError):
            artifact_report.kernel_table([], [], "triton", 0)

    def test_provenance_summary_synthetic_and_empty(self):
        rows = artifact_report.kernel_table(SYNTHETIC_EVENTS, ["axpby_kernel"], "triton", 2)
        summary = artifact_report.provenance_summary(rows)
        self.assertEqual(summary["totals_us"], {"agent": 6.0, "library": 0.0,
                                                "torch": 1.5, "memory": 3.0})
        self.assertAlmostEqual(sum(summary["shares"].values()), 1.0)
        self.assertAlmostEqual(summary["provenance"], 6.0 / 7.5)
        empty = artifact_report.provenance_summary([])
        self.assertIsNone(empty["provenance"])
        self.assertTrue(all(v is None for v in empty["shares"].values()))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(self.tmp.name)

    def report(self, **extra):
        return artifact_report.build_report(
            run_id="r1", task_id="t", backend="triton", started_at="s", finished_at="f",
            workdir=self.task, **extra)

    def test_report_round_trips_strict_json(self):
        rows = artifact_report.kernel_table(SYNTHETIC_EVENTS, ["axpby_kernel"], "triton", 2)
        report = self.report(rows=rows, capabilities=artifact_report.tool_capabilities(
            which=lambda tool: None, cupti_events=5))
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual(report["schema_version"], artifact_report.SCHEMA_VERSION)
        self.assertEqual(report["benchmark"], {"enabled": False})
        self.assertEqual(report["trace_reason"], "not captured")
        with self.assertRaises(ValueError):
            self.report(inputs=[{"value": float("nan")}])

    def test_source_hashes_track_kernel_changes(self):
        before = artifact_report.source_hashes(self.task)
        for key in ("task/model.py", "task/model_new.py", "harness/utils/rollout_env.py",
                    "harness/utils/artifact_report.py"):
            self.assertRegex(before[key], r"^[0-9a-f]{64}$")
        kernel = next(k for k in before if k.startswith("task/kernels/") and k.endswith(".py"))
        path = self.task / kernel[len("task/"):]
        path.write_text(path.read_text() + "\n# changed\n")
        (self.task / "kernels" / "__pycache__").mkdir(exist_ok=True)
        (self.task / "kernels" / "__pycache__" / "x.pyc").write_bytes(b"cache")
        after = artifact_report.source_hashes(self.task)
        self.assertNotEqual(before[kernel], after[kernel])
        self.assertEqual({k: v for k, v in before.items() if k != kernel},
                         {k: v for k, v in after.items() if k != kernel})
        (self.task / "model_new.py").unlink()
        self.assertIsNone(artifact_report.source_hashes(self.task)["task/model_new.py"])

    def test_capabilities_unavailable_and_cupti_states(self):
        caps = artifact_report.tool_capabilities(which=lambda tool: "/x/nsys" if tool == "nsys" else None)
        self.assertEqual(set(caps), set(artifact_report.TOOLS) | {"cupti"})
        self.assertEqual(caps["nsys"], {"available": True, "path": "/x/nsys"})
        self.assertEqual(caps["ncu"], {"available": False, "reason": "not found on PATH"})
        self.assertFalse(caps["cupti"]["available"])
        self.assertFalse(artifact_report.tool_capabilities(cupti_events=0)["cupti"]["available"])
        self.assertTrue(artifact_report.tool_capabilities(cupti_events=3)["cupti"]["available"])

    def test_helpers_import_without_torch(self):
        code = ("from utils import artifact_report, profile_artifacts; import sys; "
                "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


class RolloutDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(self.tmp.name)
        self.cfg = dataclasses.replace(RolloutConfig.preset("grpo"), diagnostics=True)

    def real_diagnostics(self, canned):
        canned = iter(canned)
        def fake(module, args, workdir, env, deadline):
            if module == "utils.profile_artifacts":
                return _stage(module, args, workdir, env, deadline)
            return next(canned)
        return fake

    def test_cpu_run_writes_truthful_unavailable_report(self):
        with patch.dict(os.environ, {"CUDA_AGENT_DEVICE": "cpu"}), patch(
                "utils.rollout_env._stage", side_effect=self.real_diagnostics(canned_stages())):
            result = run_task(str(self.task), self.cfg)
        self.assertEqual(set(result), DEFAULT_RESULT_KEYS | ARTIFACT_KEYS)
        assert_flat(self, result)
        self.assertEqual((result["reward/value"], result["reward/reason"]), (3, "faster_than_both"))
        self.assertEqual(result["artifact/unavailable_reason"], "CUDA_AGENT_DEVICE=cpu")
        self.assertIsNone(result["artifact/trace"])
        report_path, kernels_path = Path(result["artifact/report"]), Path(result["artifact/kernels"])
        self.assertTrue(report_path.is_file() and kernels_path.is_file())
        artifact_dir = report_path.parent
        self.assertEqual(artifact_dir.parent, self.task / ".artifacts")
        self.assertFalse((artifact_dir / "trace.json").exists())  # No fabricated trace.
        report = json.loads(report_path.read_text())
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual(report["run_id"], artifact_dir.name)
        self.assertIsNone(report["gpu"])
        self.assertEqual(report["gpu_reason"], "CUDA_AGENT_DEVICE=cpu")
        self.assertIsNone(report["trace"])
        self.assertIsNone(report["capture"])
        self.assertIsNone(report["provenance"])
        self.assertFalse(report["capabilities"]["cupti"]["available"])
        self.assertEqual(report["benchmark"], {"enabled": False})
        self.assertIn("task/model_new.py", report["source_sha256"])
        self.assertIsNone(report["software"]["cuda_runtime"])
        if HAS_TORCH:
            from importlib import metadata
            self.assertEqual(report["inputs"], [{"shape": [1, 128], "dtype": "torch.float32"}] * 2)
            self.assertEqual(report["software"]["torch"], metadata.version("torch"))
        else:
            self.assertEqual(report["inputs_reason"], "torch is not installed")
            self.assertIsNone(report["software"]["torch"])
        kernels = json.loads(kernels_path.read_text())
        self.assertEqual((kernels["measured"], kernels["kernels"]), (False, []))

    def test_argv_benchmark_samples_and_embedding(self):
        with patch.dict(os.environ, {"CUDA_AGENT_DEVICE": "cpu"}):
            cfg = dataclasses.replace(self.cfg, benchmark=True)
            calls = []
            def fake(module, args, workdir, env, deadline):
                calls.append((module, args))
                if module in ("utils.benchmark", "utils.profile_artifacts"):
                    return _stage(module, args, workdir, env, deadline)
                return canned_stages()[len(calls) - 1]
            with patch("utils.rollout_env._stage", side_effect=fake):
                result = run_task(str(self.task), cfg)
        modules = [module for module, _ in calls]
        self.assertEqual(modules[3:], ["utils.benchmark", "utils.profile_artifacts"])
        bench_args, diag_args = calls[3][1], calls[4][1]
        samples = bench_args[bench_args.index("--samples-json") + 1]
        self.assertEqual(diag_args[diag_args.index("--bench-samples") + 1], samples)
        self.assertEqual(Path(samples).parent, Path(result["artifact/report"]).parent)
        report = json.loads(Path(result["artifact/report"]).read_text())
        self.assertTrue(report["benchmark"]["enabled"])
        self.assertEqual(report["benchmark"]["summary"],
                         {"bench/available": False, "bench/unavailable_reason": "CUDA_AGENT_DEVICE=cpu"})
        self.assertFalse(result["bench/available"])
        assert_flat(self, result)

    def test_stage_failures_never_change_reward_or_trust_child_paths(self):
        outcomes = (((1, "boom", "err", False), "diagnostics stage exited 1"),
                    ((None, "", "", True), "timeout"),
                    ((0, "not json", "", False), "malformed diagnostics output"),
                    ((0, json.dumps({"artifact/report": "/etc/passwd",
                                     "artifact/unavailable_reason": None}), "", False),
                     "report.json not written"))
        for outcome, reason in outcomes:
            with self.subTest(reason=reason), patch(
                    "utils.rollout_env._stage", side_effect=canned_stages() + [outcome]):
                result = run_task(str(self.task), self.cfg)
            self.assertEqual((result["reward/value"], result["reward/reason"]), (3, "faster_than_both"))
            self.assertEqual(result["artifact/unavailable_reason"], reason)
            self.assertTrue(all(result[k] is None for k in ("artifact/report", "artifact/trace",
                                                            "artifact/kernels")))
            assert_flat(self, result)

    def test_not_run_when_an_earlier_stage_fails(self):
        with patch("utils.rollout_env._stage",
                   side_effect=canned_stages()[:1] + [(1, "x", "y", False)]) as stage:
            result = run_task(str(self.task), self.cfg)
        self.assertEqual(stage.call_count, 2)
        self.assertEqual(result["reward/reason"], "verification")
        self.assertEqual(result["artifact/unavailable_reason"], "not run")
        self.assertEqual(set(result), DEFAULT_RESULT_KEYS | ARTIFACT_KEYS)

    def test_provenance_shares_are_flat_and_null_when_unmeasured(self):
        profile = json.loads(canned_stages()[2][1].splitlines()[-1])
        profile.update({"provenance/agent": .9, "provenance/library": 0.0,
                        "provenance/torch": .05, "provenance/memory": .05})
        stages = canned_stages()[:2] + [(0, json.dumps(profile), "", False)]
        with patch("utils.rollout_env._stage", side_effect=stages):
            result = run_task(str(self.task), RolloutConfig())
        self.assertEqual((result["env/provenance_agent"], result["env/provenance_memory"]), (.9, .05))
        self.assertEqual(set(result), DEFAULT_RESULT_KEYS)
        with patch("utils.rollout_env._stage", side_effect=canned_stages()):
            result = run_task(str(self.task), RolloutConfig())
        self.assertTrue(all(result["env/provenance_" + s] is None
                            for s in ("agent", "library", "torch", "memory")))

    def test_artifact_keys_are_dropped_from_wandb_rows(self):
        from tracking import wandb_log
        row = wandb_log._case_row("c", {"reward/value": 1, "artifact/report": "/p",
                                        "log/stderr_tail": "x"})
        self.assertFalse(any(k.startswith(("artifact/", "log/")) for k in row))


@unittest.skipUnless(HAS_GPU and HAS_TRITON,
                     "requires a CUDA GPU; the diagnostic trace and kernel table never ran here")
class GPUArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)

    def test_trace_kernel_table_and_report(self):
        task = make_task(self.tmp.name, "triton")
        cfg = dataclasses.replace(RolloutConfig.preset("grpo"), diagnostics=True, benchmark=True)
        result = run_task(str(task), cfg)
        keep = os.environ.get("CUDA_AGENT_KEEP_PROFILE_ARTIFACTS")
        if keep and result.get("artifact/report"):
            # Opt-in evidence copy (the Modal runner sets it); scratch is deleted.
            import shutil
            source = Path(result["artifact/report"]).parent
            shutil.copytree(source, Path(keep) / source.name, dirs_exist_ok=True)
            (Path(keep) / source.name / "rollout-result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n")
        assert_flat(self, result)
        self.assertIsNone(result["artifact/unavailable_reason"], result)
        for key in ("artifact/report", "artifact/trace", "artifact/kernels"):
            self.assertTrue(Path(result[key]).is_file(), key)
        trace = json.loads(Path(result["artifact/trace"]).read_text())
        self.assertTrue(trace.get("traceEvents"))
        kernels = json.loads(Path(result["artifact/kernels"]).read_text())
        self.assertTrue(kernels["measured"])
        self.assertIn("agent", {row["class"] for row in kernels["kernels"]})
        self.assertTrue(all(math.isfinite(row["total_us"]) and row["total_us"] >= 0
                            for row in kernels["kernels"]))
        report = json.loads(Path(result["artifact/report"]).read_text())
        self.assertTrue(report["gpu"]["name"])
        self.assertTrue(report["capabilities"]["cupti"]["available"])
        self.assertTrue(report["benchmark"]["summary"]["bench/available"])
        self.assertEqual(report["software"]["torch"], torch.__version__)
        self.assertGreater(report["provenance"]["shares"]["agent"], 0)


if __name__ == "__main__":
    unittest.main()
