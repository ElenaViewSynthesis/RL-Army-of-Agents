"""--cuda-graphs is opt-in, needs the benchmark, and can never change a reward.

Two promises from the README's "CUDA-graph replay" section, pinned here:
1. Every entry point requires the benchmark for --cuda-graphs, instead of
   silently ignoring it.
2. Graph results (capturable, replay correctness, replay timings) are
   diagnostics. Making graph-capturability part of the agent's contract, or
   using graph timings in the reward, needs the user's approval, so the
   reward code must not read them. All stage outputs here are SYNTHETIC.
"""

import dataclasses
import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import ROOT, make_task, scratch
from utils import reward, task_policy
from utils.config import RolloutConfig
from utils.rollout_env import run_task

REWARD_KEYS = ("reward/value", "reward/base", "reward/reason", "reward/overlong_penalty")


def stages(bench_record=None):
    profile = {"timing_measured": True, "t_eager_us": 100.0, "t_compile_us": 80.0,
               "t_gen_us": 50.0, "provenance": 1.0}
    out = [(0, "Compile success\n", "", False),
           (0, "[PASS] check 5/5\n[PASS] verify success\n", "", False),
           (0, json.dumps(profile), "", False)]
    if bench_record is not None:
        out.append((0, json.dumps(bench_record), "", False))
    return out


GRAPH_GOOD = {"bench/available": True, "bench/generated/graph_capturable": True,
              "bench/generated/graph_replay_ok": True, "bench/generated/graph_us_median": 0.001,
              "bench/graph_all_capturable": True}
GRAPH_BAD = {"bench/available": True, "bench/generated/graph_capturable": False,
             "bench/generated/graph_error": "RuntimeError: operation not permitted when stream is capturing",
             "bench/generated/graph_replay_ok": False, "bench/generated/graph_us_median": 1e9,
             "bench/graph_all_capturable": False}


class RequiresBenchmarkTests(unittest.TestCase):
    def test_rollout_config_refuses_graphs_without_benchmark(self):
        with self.assertRaisesRegex(ValueError, "cuda_graphs requires benchmark"):
            RolloutConfig(cuda_graphs=True)
        RolloutConfig(benchmark=True, cuda_graphs=True)
        with self.assertRaises(ValueError):
            RolloutConfig.from_args({"cuda_graphs": True})
        self.assertTrue(RolloutConfig.from_args({"cuda_graphs": True, "benchmark": True}).cuda_graphs)

    def test_rollout_cli_refuses_graphs_without_benchmark(self):
        with scratch() as tmp:
            task = make_task(tmp)
            proc = subprocess.run([sys.executable, "-S", "-m", "utils.rollout_env", str(task), "--cuda-graphs"],
                                  cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("cuda_graphs requires benchmark", proc.stderr)

    def test_validate_cli_refuses_graphs_without_benchmark_or_suite(self):
        for argv, message in ((["--suite", "stage6", "--cuda-graphs"], "--cuda-graphs requires --benchmark"),
                              (["--benchmark", "--cuda-graphs"], "--cuda-graphs applies only to --suite runs")):
            with self.subTest(argv=argv):
                proc = subprocess.run([sys.executable, "-S", "-m", "scripts.validate", *argv],
                                      cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn(message, proc.stderr)


class TaskSuiteGraphConfigTests(unittest.TestCase):
    def test_every_fixture_gets_a_valid_config_with_cuda_graphs(self):
        # Regression: run stage6-acceptance-20261111T200022Z aborted after one row
        # because non-honest fixtures got cuda_graphs=True with benchmark=False.
        from utils.task_catalog import select_cases
        from utils.task_suite import VARIANTS, evaluate_case
        seen = {}

        def fake_run(workdir, cfg, *args, **kwargs):
            seen[Path(workdir).name] = (cfg.benchmark, cfg.cuda_graphs)
            return {"env/verify_ok": False, "reward/value": -1.0, "reward/base": -1,
                    "reward/reason": "verification"}

        requested = RolloutConfig(benchmark=True, cuda_graphs=True)
        case = select_cases("lora_shrink_bgmv", "smoke")[0]
        with scratch() as tmp, patch("utils.task_suite.cuda_unavailable", return_value=None),                 patch("utils.task_suite.available_backends", return_value={"triton": None}),                 patch("utils.task_suite.run_task", side_effect=fake_run),                 patch("utils.task_suite._stage", return_value=(1, "", "", False)):
            for variant in VARIANTS:
                evaluate_case("lora_shrink_bgmv", case.id, "triton", variant, Path(tmp) / variant,
                              cfg=requested)
        self.assertEqual(seen["honest"], (True, True))
        for variant in set(VARIANTS) - {"honest"}:
            self.assertEqual(seen[variant], (False, False), variant)


class RewardNeverReadsGraphsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.task = make_task(self.tmp.name)

    def rewards(self, cfg, bench_record):
        with patch("utils.rollout_env._stage", side_effect=stages(bench_record)):
            result = run_task(str(self.task), cfg)
        return {key: result[key] for key in REWARD_KEYS}, result

    def test_run_task_reward_is_identical_whatever_the_graph_results(self):
        base, _ = self.rewards(RolloutConfig(), None)
        cfg = RolloutConfig(benchmark=True, cuda_graphs=True)
        good, good_result = self.rewards(cfg, GRAPH_GOOD)
        bad, bad_result = self.rewards(cfg, GRAPH_BAD)
        self.assertEqual(base, good)
        self.assertEqual(base, bad)
        self.assertEqual(base["reward/reason"], "faster_than_both")
        # The graph results are still recorded, as diagnostics.
        self.assertFalse(bad_result["bench/generated/graph_capturable"])
        self.assertTrue(good_result["bench/graph_all_capturable"])

    def test_task_policy_ignores_graph_results(self):
        with patch("utils.rollout_env._stage", side_effect=stages(None)):
            profile = {}
            raw = run_task(str(self.task), RolloutConfig(), profile_out=profile)
        profile.update({f"provenance/{k}": v for k, v in
                        (("agent", 1.0), ("library", 0.0), ("torch", 0.0), ("memory", 0.0))})
        cfg = RolloutConfig(benchmark=True, cuda_graphs=True)
        plain = task_policy.apply_task_policy(raw, profile, cfg, turns_used=1)
        for record in (GRAPH_GOOD, GRAPH_BAD):
            with self.subTest(capturable=record["bench/generated/graph_capturable"]):
                graded = task_policy.apply_task_policy({**raw, **record}, profile, cfg, turns_used=1)
                self.assertEqual({k: graded[k] for k in REWARD_KEYS}, {k: plain[k] for k in REWARD_KEYS})

    def test_reward_functions_take_no_benchmark_or_graph_input(self):
        self.assertEqual(list(inspect.signature(reward.compute_reward).parameters),
                         ["verify_ok", "t_eager", "t_compile", "t_gen", "provenance", "cfg", "turns_used"])
        # The functions that set rewards; task_policy.diagnostic_rates only writes
        # throughput diagnostics and is deliberately not included.
        for function in (reward.compute_reward, task_policy.apply_task_policy):
            source = inspect.getsource(function)
            with self.subTest(function=function.__qualname__):
                self.assertNotIn("graph", source.lower())
                self.assertNotIn("bench", source.lower())


if __name__ == "__main__":
    unittest.main()
