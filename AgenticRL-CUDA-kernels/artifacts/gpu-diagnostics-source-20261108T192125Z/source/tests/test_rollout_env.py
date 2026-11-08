import dataclasses
import json
import os
import subprocess
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from tests.support import ROOT, assert_flat, make_task, scratch
from utils.config import RolloutConfig
from utils.rollout_env import _stage, run_task


def profile_output(provenance=1.0):
    # Synthetic subprocess protocol fixture, never reported as a GPU measurement.
    return 'Torch Baseline: 100.000us, Torch Compile: 80.000us, CUDA Extension: 50.000us\n' + json.dumps({
        "timing_measured": True, "t_eager_us": 100.0, "t_compile_us": 80.0,
        "t_gen_us": 50.0, "provenance": provenance})


class RolloutProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(self.tmp.name)
        self.cfg = RolloutConfig.preset("grpo")

    def stages(self, provenance=1.0):
        return [(0, "Compile success\n", "", False),
                (0, "[PASS] check 5/5\n[PASS] verify success\n", "", False),
                (0, profile_output(provenance), "", False)]

    def test_success_flat_schema_deadline_and_visibility(self):
        before = dict(os.environ)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), patch(
                "utils.rollout_env._stage", side_effect=self.stages()) as stage:
            times = {"stale": -1}
            result = run_task(str(self.task), self.cfg, 1, stage_times_out=times)
            self.assertEqual(set(times), {"compile", "verification", "profiling"})
            self.assertTrue(all(t >= 0 for t in times.values()))
            calls = stage.call_args_list
            self.assertTrue(all(c.args[3]["CUDA_VISIBLE_DEVICES"] == "3" for c in calls))
            self.assertEqual(len({c.args[4] for c in calls}), 1)
            self.assertEqual(calls[2].args[1], ["--backend", "triton", "--iters", "10", "--seed", "0", "--json"])
        self.assertEqual(dict(os.environ), before)
        self.assertEqual((result["reward/value"], result["reward/base"], result["meta/backend"]), (3, 3, "triton"))
        self.assertEqual(result["env/speedup_vs_compile"], 1.6)
        self.assertEqual(len(result), 18)
        assert_flat(self, result)

    def test_benchmark_stage_opt_in(self):
        expected_keys = {"env/verify_ok", "env/t_eager_us", "env/t_compile_us",
                         "env/t_gen_us", "env/speedup_vs_compile", "env/provenance",
                         "env/wall_s", "env/turns_used", "reward/value", "reward/base",
                         "reward/reason", "reward/overlong_penalty", "meta/backend",
                         "meta/task_id", "meta/algo", "meta/seed",
                         "log/compile_tail", "log/stderr_tail"}
        with patch("utils.rollout_env._stage", side_effect=self.stages()) as stage:
            result = run_task(str(self.task), self.cfg)
        self.assertEqual(stage.call_count, 3)
        self.assertEqual(set(result), expected_keys)
        bench_stdout = json.dumps({"bench/available": True,
                                   "bench/generated/event_us_median": 12.5,
                                   "bench/device_name": "Fake"})
        cfg = dataclasses.replace(self.cfg, benchmark=True)
        stages = self.stages() + [(0, bench_stdout, "", False)]
        with patch("utils.rollout_env._stage", side_effect=stages) as stage:
            result = run_task(str(self.task), cfg)
        self.assertEqual(stage.call_count, 4)
        self.assertEqual(stage.call_args_list[3].args[1],
                         ["--backend", "triton", "--warmup-iters", "10",
                          "--rounds", "6", "--iters-per-round", "10",
                          "--seed", "0", "--json"])
        self.assertEqual(set(result), expected_keys | {"bench/available",
                                                       "bench/generated/event_us_median",
                                                       "bench/device_name"})
        self.assertEqual((result["reward/value"], result["reward/base"]), (3, 3))
        assert_flat(self, result)
        for outcome, unavailable in (((1, "boom", "err", False), "benchmark stage exited 1"),
                                     ((None, "", "", True), "timeout"),
                                     ((0, "not json", "", False), "malformed benchmark output")):
            stages = self.stages() + [outcome]
            with self.subTest(unavailable=unavailable), patch(
                    "utils.rollout_env._stage", side_effect=stages):
                result = run_task(str(self.task), cfg)
            # A failed benchmark stage degrades to unavailable, never a reward.
            self.assertEqual((result["reward/value"], result["reward/base"]), (3, 3))
            self.assertEqual(result["reward/reason"], "faster_than_both")
            self.assertFalse(result["bench/available"])
            self.assertEqual(result["bench/unavailable_reason"], unavailable)
            if unavailable == "benchmark stage exited 1":
                self.assertIn("err", result["log/stderr_tail"])
                self.assertIn("boom", result["log/stderr_tail"])
            assert_flat(self, result)
        cfg = dataclasses.replace(self.cfg, benchmark=True, cuda_graphs=True)
        with patch("utils.rollout_env._stage", side_effect=self.stages() + [(0, bench_stdout, "", False)]) as stage:
            run_task(str(self.task), cfg)
        self.assertIn("--cuda-graphs", stage.call_args_list[3].args[1])

    def test_provenance_gate_and_overlong_via_entry_point(self):
        with patch("utils.rollout_env._stage", side_effect=self.stages(.1)):
            hack = run_task(str(self.task), self.cfg)
        self.assertTrue(hack["env/verify_ok"])
        self.assertEqual((hack["reward/value"], hack["reward/reason"]), (-1, "provenance"))
        for algo, penalty in (("grpo", 0), ("dapo", .75), ("cispo", .75)):
            with patch("utils.rollout_env._stage", side_effect=self.stages()):
                result = run_task(str(self.task), RolloutConfig.preset(algo), 19)
            self.assertEqual(result["reward/base"], 3)
            self.assertEqual(result["reward/overlong_penalty"], penalty)
            self.assertEqual(result["reward/value"], 3 - penalty)
            assert_flat(self, result)

    def test_max_turns_skips_subprocesses(self):
        with patch("utils.rollout_env._stage") as stage:
            result = run_task(str(self.task), self.cfg, 25)
        stage.assert_not_called()
        self.assertEqual(result["reward/reason"], "max_turns")
        self.assertEqual(result["reward/value"], -1)
        assert_flat(self, result)

    def test_generated_syntax_error_is_a_result_not_a_parent_crash(self):
        (self.task / "model_new.py").write_text("def unfinished(:\n")
        result = run_task(str(self.task), self.cfg)
        self.assertEqual(result["reward/value"], -1)
        self.assertEqual(result["reward/reason"], "backend")
        self.assertIn("SyntaxError", result["log/stderr_tail"])
        assert_flat(self, result)

    def test_stage_failures_and_timeout(self):
        for i, reason in enumerate(("compile", "verification", "profiling")):
            for timeout in (False, True):
                stages = self.stages()[:i] + [(1, "failed", "a useful error", timeout)]
                with self.subTest(stage=reason, timeout=timeout), patch("utils.rollout_env._stage", side_effect=stages):
                    result = run_task(str(self.task), self.cfg)
                self.assertEqual(result["reward/reason"], "timeout" if timeout else reason)
                self.assertEqual(result["reward/value"], -1)
                self.assertIn("useful error", result["log/stderr_tail"])
                assert_flat(self, result)

    def test_malformed_profile_missing_workdir_and_unmeasured(self):
        for output in ("invalid-json", "{}", profile_output(float("nan"))):
            stages = self.stages()[:2] + [(0, output, "", False)]
            with patch("utils.rollout_env._stage", side_effect=stages):
                result = run_task(str(self.task), self.cfg)
            self.assertEqual(result["reward/reason"], "profiling")
            assert_flat(self, result)
        self.assertEqual(run_task(str(self.task / "missing"), self.cfg)["reward/reason"], "workdir")
        stages = self.stages()[:2] + [(0, json.dumps({"t_eager_us": None, "t_compile_us": None,
                                                  "t_gen_us": None, "provenance": None}), "", False)]
        with patch("utils.rollout_env._stage", side_effect=stages):
            result = run_task(str(self.task), self.cfg)
        self.assertTrue(result["env/verify_ok"])
        self.assertEqual(result["reward/reason"], "unmeasured")
        assert_flat(self, result)

    def test_cli_without_torch_honors_preset_and_cap(self):
        proc = subprocess.run([sys.executable, "-S", "-m", "utils.rollout_env", str(self.task),
                               "--algo", "dapo", "--turns-used", "25", "--max-turns", "20"],
                              cwd=ROOT, capture_output=True, text=True, check=True)
        result = json.loads(proc.stdout)
        self.assertEqual(result["meta/algo"], "dapo")
        self.assertEqual(result["reward/reason"], "max_turns")
        assert_flat(self, result)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / ".cache").mkdir()

    def test_cleanup_wait_is_bounded_even_if_kill_cannot_reap(self):
        proc = Mock(pid=12345, returncode=None)
        proc.wait.side_effect = [subprocess.TimeoutExpired("worker", 1),
                                 subprocess.TimeoutExpired("worker", 1)]
        with patch("utils.rollout_env.subprocess.Popen", return_value=proc), patch("utils.rollout_env.os.killpg"):
            _, _, _, timeout = _stage("unused", [], self.path, os.environ.copy(), time.monotonic() + 1)
        self.assertTrue(timeout)
        self.assertEqual(len(proc.wait.call_args_list), 2)
        self.assertTrue(all(call.kwargs.get("timeout") is not None for call in proc.wait.call_args_list))

    def test_task_namespace_kernels_and_trusted_utils(self):
        # Upstream tasks have no kernels/__init__.py. The harness's regular
        # kernels package must never shadow a task's namespace package.
        (self.path / "kernels").mkdir()
        (self.path / "kernels/custom.py").write_text('VALUE = "task-specific-kernel"\n')
        (self.path / "utils").mkdir()
        (self.path / "utils/__init__.py").write_text('raise RuntimeError("untrusted utils imported")\n')
        (self.path / "namespace_check.py").write_text(
            'from kernels.custom import VALUE\nimport utils\nprint(VALUE)\nprint(utils.__file__)\n')
        code, stdout, stderr, timeout = _stage("namespace_check", [], self.path, os.environ.copy(), time.monotonic() + 5)
        self.assertEqual(code, 0, stderr)
        self.assertFalse(timeout)
        self.assertIn("task-specific-kernel", stdout)
        self.assertIn(str(ROOT / "utils/__init__.py"), stdout)

    def test_real_timeout_kills_grandchild_and_preserves_output(self):
        (self.path / "sleeper.py").write_text(
            'import pathlib, subprocess, sys, time\n'
            'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
            'pathlib.Path("child.pid").write_text(str(child.pid))\n'
            'print("sleeping now", flush=True)\ntime.sleep(60)\n')
        start = time.monotonic()
        code, stdout, stderr, timeout = _stage("sleeper", [], self.path, os.environ.copy(), start + 2)
        self.assertTrue(timeout)
        self.assertLess(time.monotonic() - start, 5)
        self.assertIn("sleeping now", stdout)
        pid = int((self.path / "child.pid").read_text())
        status = Path(f"/proc/{pid}/stat")
        # A zombie has already been killed; PID 1 may reap it asynchronously.
        for _ in range(20):
            if not status.exists() or status.read_text().split()[2] in ("Z", "X"):
                break
            time.sleep(.05)
        else:
            self.fail("Timed-out grandchild is still executing")

    def test_crash_isolated_and_concurrent_workdirs(self):
        (self.path / "crasher.py").write_text('import os\nos._exit(7)\n')
        code, _, _, timeout = _stage("crasher", [], self.path, os.environ.copy(), time.monotonic() + 5)
        self.assertEqual(code, 7)
        self.assertFalse(timeout)
        tasks = []
        for name in ("a", "b"):
            path = self.path / name
            path.mkdir()
            (path / ".cache").mkdir()
            (path / "echo_task.py").write_text('from pathlib import Path\nprint(Path.cwd().name)\n')
            tasks.append(path)
        def evaluate(path):
            return _stage("echo_task", [], path, os.environ.copy(), time.monotonic() + 5)[1].strip()
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(evaluate, tasks)), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
