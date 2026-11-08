"""Diagnostic orchestration tests: fake processes only; never run a GPU or a vendor tool."""

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from tests.support import ROOT, scratch
from utils import diagnose


def staged(self, tool, records, *, gpu=None, worker=None, captures=(), kernel=None):
    from unittest.mock import patch
    calls = []

    def fake(argv, workdir, env, deadline, out, label):
        calls.append(label)
        returncode, timed_out = records[len(calls) - 1]
        stage = {"argv": argv, "command": "fake", "stdout": str(out / f"{label}.stdout.log"),
                 "stderr": str(out / f"{label}.stderr.log"), "returncode": returncode,
                 "timed_out": timed_out, "wall_s": 0.01}
        Path(stage["stdout"]).write_text("fake tool 1.0\n")
        Path(stage["stderr"]).write_text("")
        if label == "gpu" and gpu is not None:
            (out / "gpu.json").write_text(json.dumps(gpu))
        if label == "tool":
            if worker is not None:
                (out / "worker.json").write_text(json.dumps(worker))
            for name in captures:
                (out / name).write_bytes(b"raw")
        return stage

    with scratch() as tmp:
        tmp = Path(tmp)
        with patch.object(diagnose, "run_process", side_effect=fake), \
             patch.object(diagnose.shutil, "which", return_value="/fake/tool"):
            result = diagnose.run_diagnostic(tool, tmp, tmp / "out", backend="triton",
                                             kernel=kernel)
        report = json.loads((tmp / "out" / "report.json").read_text())
        report["review_json_exists"] = (tmp / "out" / "review.json").is_file()
        return result, report, calls


class RunDiagnosticTests(unittest.TestCase):
    GPU = {"available": True, "completed": True, "gpu_uuid": "gpu-u", "gpu_name": "fake-gpu"}
    WORKER = {"diagnostic_only": True, "completed": True, "mode": "verify", "backend": "triton"}

    def test_missing_binary_is_unavailable_without_processes(self):
        from unittest.mock import patch
        with scratch() as tmp:
            tmp = Path(tmp)
            with patch.object(diagnose.shutil, "which", return_value=None), \
                 patch.object(diagnose, "run_process", side_effect=AssertionError("must not exec")):
                result = diagnose.run_diagnostic("sanitizer", tmp, tmp / "out", backend="triton")
            report = json.loads((tmp / "out" / "report.json").read_text())
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("unavailable", "tool_not_installed"))
        self.assertEqual(report["stages"], [])
        self.assertIsNone(result["diag/reported_errors"])
        self.assertIsNone(result["diag/kernel_count"])

    def test_cpu_device_is_unavailable_without_processes(self):
        from unittest.mock import patch
        env = {**os.environ, "CUDA_AGENT_DEVICE": "cpu"}
        with scratch() as tmp:
            tmp = Path(tmp)
            with patch.dict("os.environ", env, clear=True), \
                 patch.object(diagnose.shutil, "which", return_value="/fake/tool"), \
                 patch.object(diagnose, "run_process", side_effect=AssertionError("must not exec")):
                result = diagnose.run_diagnostic("sanitizer", tmp, tmp / "out", backend="triton")
            report = json.loads((tmp / "out" / "report.json").read_text())
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("unavailable", "cuda_required"))
        self.assertEqual(report["stages"], [])

    def test_negative_control_and_kernel_restrictions(self):
        with scratch() as tmp:
            tmp = Path(tmp)
            for i, kwargs in enumerate(({"tool": "nsys"}, {"tool": "ncu"},
                                        {"tool": "sanitizer", "check": "racecheck"},
                                        {"tool": "sanitizer", "backend": "cuda"})):
                with self.assertRaises(ValueError):
                    diagnose.run_diagnostic(kwargs["tool"], tmp, tmp / f"o{i}",
                                            backend=kwargs.get("backend", "triton"),
                                            check=kwargs.get("check", "memcheck"),
                                            negative_control=True)
            with self.assertRaises(ValueError):
                diagnose.run_diagnostic("sanitizer", tmp, tmp / "k", backend="triton", kernel="x")
            with self.assertRaises(ValueError):
                diagnose.run_diagnostic("sanitizer", tmp, tmp / "t", backend="triton",
                                        timeout_s=float("nan"))
        from unittest.mock import patch
        with scratch() as tmp:
            tmp = Path(tmp)
            with patch.object(diagnose.shutil, "which", return_value=None):
                result = diagnose.run_diagnostic("ncu", tmp, tmp / "nc",
                                                 backend="triton", kernel="axpby.*")
            self.assertEqual(result["diag/status"], "unavailable")

    def test_tool_failure_timeout_and_incomplete_worker_fail_closed(self):
        for records, status, reason in (
                ([(0, False), (0, False), (1, False)], "failed", "tool_failed"),
                ([(0, False), (0, False), (0, True)], "timeout", "tool_timeout")):
            result, _, calls = staged(self, "sanitizer", records, gpu=self.GPU,
                                      worker=self.WORKER)
            self.assertEqual((result["diag/status"], result["diag/reason"]), (status, reason))
            self.assertEqual(calls, ["version", "gpu", "tool"])
            self.assertFalse(result["diag/evidence_reviewed"])
        result, _, _ = staged(self, "sanitizer", [(0, False), (0, False), (0, False)],
                              gpu=self.GPU, worker={**self.WORKER, "completed": False})
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("failed", "worker_incomplete"))

    def test_captured_status_is_unreviewed_with_null_counts(self):
        result, report, _ = staged(self, "sanitizer", [(0, False), (0, False), (0, False)],
                                   gpu=self.GPU, worker=self.WORKER)
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("captured", "raw_evidence_requires_review"))
        self.assertFalse(result["diag/evidence_reviewed"])
        self.assertTrue(result["diag/diagnostic_only"])
        self.assertIsNone(result["diag/reported_errors"])
        self.assertIsNone(result["diag/kernel_count"])
        self.assertEqual(result["diag/gpu_uuid"], "gpu-u")
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertTrue(all(k.startswith("diag/") for k in result))

    def test_nsys_requires_trace_files(self):
        result, _, _ = staged(self, "nsys", [(0, False), (0, False), (0, False)],
                              gpu=self.GPU, worker=self.WORKER)
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("failed", "missing_trace"))
        result, report, _ = staged(self, "nsys", [(0, False), (0, False), (0, False)],
                                   gpu=self.GPU, worker=self.WORKER,
                                   captures=("capture.nsys-rep", "capture.sqlite"))
        self.assertEqual(result["diag/status"], "captured")
        self.assertTrue(report["review_json_exists"])

    def test_ncu_requires_report_then_metrics_export(self):
        result, _, calls = staged(self, "ncu", [(0, False), (0, False), (0, False)],
                                  gpu=self.GPU, worker=self.WORKER, kernel="axpby.*")
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("failed", "missing_counter_report"))
        result, _, calls = staged(self, "ncu", [(0, False), (0, False), (0, False), (1, False)],
                                  gpu=self.GPU, worker=self.WORKER,
                                  captures=("capture.ncu-rep",), kernel="axpby.*")
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("failed", "metrics_export_failed"))
        self.assertEqual(calls, ["version", "gpu", "tool", "metrics"])
        result, _, _ = staged(self, "ncu", [(0, False), (0, False), (0, False), (0, False)],
                              gpu=self.GPU, worker=self.WORKER,
                              captures=("capture.ncu-rep",), kernel="axpby.*")
        self.assertEqual(result["diag/status"], "captured")


class EnvironmentTests(unittest.TestCase):
    def test_child_environment_allowlist_strips_secrets_keeps_cuda(self):
        from unittest.mock import patch
        env = {"PATH": os.environ["PATH"], "HOME": "/h",
               "SECRET_API_KEY": "leak-me", "AWS_SECRET_ACCESS_KEY": "leak-me-too",
               "CUDA_VISIBLE_DEVICES": "3", "CUDA_AGENT_DEVICE": "cuda:0",
               "WANDB_MODE": "online", "TRITON_INTERPRET": "1"}
        with scratch() as tmp, patch.dict("os.environ", env, clear=True):
            child = diagnose.diagnostic_environment(Path(tmp), "sanitizer")
            child_nsys = diagnose.diagnostic_environment(Path(tmp), "nsys")
        self.assertNotIn("SECRET_API_KEY", child)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", child)
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(child["CUDA_AGENT_DEVICE"], "cuda:0")
        self.assertEqual(child["WANDB_MODE"], "disabled")
        self.assertEqual(child["TRITON_INTERPRET"], "0")
        self.assertEqual(child["PYTORCH_NO_CUDA_MEMORY_CACHING"], "1")
        self.assertNotIn("PYTORCH_NO_CUDA_MEMORY_CACHING", child_nsys)
        self.assertEqual(child_nsys["NSYS_NVTX_PROFILER_REGISTER_ONLY"], "0")

    def test_tool_commands(self):
        out = Path("/nonexistent/out")
        nsys = diagnose.tool_command("nsys", "nsys-bin", ["worker"], out, "memcheck", None)
        self.assertIn("--discard-environment=true", nsys)
        self.assertIn("--capture-range=nvtx", nsys)
        self.assertIn("--nvtx-capture=agenticrl_generated", nsys)
        self.assertIn("--capture-range-end=stop", nsys)
        ncu = diagnose.tool_command("ncu", "ncu-bin", ["worker"], out, "memcheck", "axpby.*")
        self.assertEqual(ncu[ncu.index("--nvtx-include") + 1], "agenticrl_generated/")
        self.assertEqual(ncu[ncu.index("--kernel-name") + 1], "regex:axpby.*")
        self.assertEqual(ncu[ncu.index("--launch-count") + 1], "1")
        san = diagnose.tool_command("sanitizer", "cs-bin", ["worker"], out, "racecheck", None)
        self.assertEqual(san[san.index("--tool") + 1], "racecheck")
        self.assertEqual(san[san.index("--error-exitcode") + 1], "86")

    def test_imports_under_python_S_are_torch_free(self):
        code = ("import utils.diagnose, utils.diagnostic_worker, utils.diagnostic_review, "
                "scripts.diagnose_gpu; import sys; "
                "assert 'torch' not in sys.modules and 'wandb' not in sys.modules")
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)

    def test_run_process_kills_timed_out_process_group(self):
        with scratch() as tmp:
            tmp = Path(tmp)
            marker = tmp / "survived.txt"
            inner = f"import time,pathlib;time.sleep(2);pathlib.Path({str(marker)!r}).write_text('x')"
            script = (f"import subprocess,sys,time;"
                      f"subprocess.Popen([sys.executable,'-c',{inner!r}]);time.sleep(60)")
            record = diagnose.run_process([sys.executable, "-c", script], tmp, dict(os.environ),
                                          time.monotonic() + 1.0, tmp, "p")
            self.assertTrue(record["timed_out"])
            time.sleep(2.5)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
