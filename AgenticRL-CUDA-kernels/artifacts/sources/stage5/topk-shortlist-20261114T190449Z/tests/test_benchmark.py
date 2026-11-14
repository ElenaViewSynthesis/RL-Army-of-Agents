"""Benchmark statistics and the standalone benchmark CLI; GPU checks skip."""

import dataclasses
import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest

from tests.support import ROOT, assert_flat, make_task, scratch
from utils import stats
from utils.config import RolloutConfig
from utils.rollout_env import run_task

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRITON = importlib.util.find_spec("triton") is not None
if HAS_TORCH:
    import torch
    HAS_GPU = torch.cuda.is_available()
else:
    HAS_GPU = False

CPU_UNAVAILABLE = {"bench/available": False,
                   "bench/unavailable_reason": "CUDA_AGENT_DEVICE=cpu"}


def bench_samples(rounds=6):
    def entry(event, wall):
        return {"event_us": list(event), "wall_us": list(wall),
                "prepare_s": 1.5, "warmup_iters": 10}
    return {"eager": entry([100.0] * rounds, [120.0] * rounds),
            "compile": entry([80.0] * rounds, [100.0] * rounds),
            "generated": entry([40.0, 50.0, 60.0] * (rounds // 3),
                               [50.0, 60.0, 70.0] * (rounds // 3))}


class StatsTests(unittest.TestCase):
    def test_summarize_known_values(self):
        s = stats.summarize([3, 1, 2])
        self.assertEqual((s["n"], s["median"], s["min"], s["max"],
                          s["p25"], s["p75"], s["cv"]), (3, 2, 1, 3, 1.5, 2.5, 0.5))
        s = stats.summarize([1, 2, 3, 4])
        self.assertEqual((s["median"], s["p25"], s["p75"]), (2.5, 1.75, 3.25))
        s = stats.summarize([7])
        self.assertEqual((s["p25"], s["p75"], s["cv"]), (7, 7, 0.0))

    def test_summarize_prefix_and_rejection(self):
        s = stats.summarize([1, 2], prefix="bench/eager/event_us_")
        self.assertEqual(s["bench/eager/event_us_n"], 2)
        for bad in ([], [float("nan")], [float("inf")], [-1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                stats.summarize(bad)

    def test_round_order_balanced_over_six(self):
        names = ("eager", "compile", "generated")
        orders = [stats.round_order(names, r) for r in range(6)]
        for position in range(3):
            for name in names:
                self.assertEqual(sum(o[position] == name for o in orders), 2)
        self.assertEqual(stats.round_order(names, 3), stats.round_order(names, 0))

    def test_summarize_bench_flat_no_graphs(self):
        out = stats.summarize_bench(bench_samples(), rounds=6)
        self.assertEqual(out["bench/generated/event_us_median"], 50.0)
        self.assertEqual(out["bench/generated/wall_us_median"], 60.0)
        self.assertEqual(out["bench/generated/event_wall_ratio"], 50.0 / 60.0)
        self.assertEqual(out["bench/generated/event_us_min"], 40.0)
        self.assertEqual(out["bench/eager/event_us_n"], 6)
        # prepare_s is reported on its own key, never inside *_us_* stats.
        self.assertEqual(out["bench/eager/prepare_s"], 1.5)
        self.assertEqual(out["bench/eager/warmup_iters"], 10)
        self.assertFalse(any("graph" in k for k in out))
        assert_flat(self, out)

    def test_summarize_bench_zero_wall_and_bad_length(self):
        samples = bench_samples()
        samples["generated"]["wall_us"] = [0.0] * 6
        out = stats.summarize_bench(samples, rounds=6)
        self.assertIsNone(out["bench/generated/event_wall_ratio"])
        samples = bench_samples()
        samples["compile"]["wall_us"] = [1.0]
        with self.assertRaises(ValueError):
            stats.summarize_bench(samples, rounds=6)

    def test_summarize_bench_graph_keys_only_when_present(self):
        samples = bench_samples()
        samples["generated"].update(graph_capturable=True, graph_error=None,
                                    graph_replay_ok=True, graph_us=[10.0] * 6)
        samples["eager"].update(graph_capturable=False, graph_error="RuntimeError: x",
                                graph_replay_ok=None, graph_us=[])
        out = stats.summarize_bench(samples, rounds=6)
        self.assertEqual((out["bench/generated/graph_capturable"],
                          out["bench/generated/graph_error"],
                          out["bench/generated/graph_replay_ok"]), (True, None, True))
        self.assertEqual(out["bench/generated/graph_us_median"], 10.0)
        self.assertEqual(out["bench/generated/graph_us_n"], 6)
        self.assertEqual((out["bench/eager/graph_capturable"],
                          out["bench/eager/graph_error"],
                          out["bench/eager/graph_replay_ok"]),
                         (False, "RuntimeError: x", None))
        self.assertNotIn("bench/eager/graph_us_median", out)
        self.assertNotIn("bench/compile/graph_capturable", out)
        assert_flat(self, out)


class CLITests(unittest.TestCase):
    def bench_cli(self, *extra):
        return subprocess.run(
            [sys.executable, "-S", "-m", "utils.benchmark", "--json", *extra],
            cwd=ROOT, env={**os.environ, "CUDA_AGENT_DEVICE": "cpu"},
            capture_output=True, text=True)

    def test_cpu_unavailable_without_torch(self):
        # python -S on a torch-free interpreter proves torch was never imported.
        proc = self.bench_cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(proc.stdout.strip().splitlines()), 1)
        self.assertEqual(json.loads(proc.stdout), CPU_UNAVAILABLE)

    def test_cpu_skip_line_without_json_flag(self):
        proc = subprocess.run(
            [sys.executable, "-S", "-m", "utils.benchmark"],
            cwd=ROOT, env={**os.environ, "CUDA_AGENT_DEVICE": "cpu"},
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(),
                         "[SKIP] benchmark unavailable: CUDA_AGENT_DEVICE=cpu")
        self.assertNotIn("{", proc.stdout)

    def test_cuda_graphs_still_unavailable_on_cpu(self):
        proc = self.bench_cli("--cuda-graphs")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), CPU_UNAVAILABLE)

    def test_rounds_must_be_a_positive_multiple_of_three(self):
        self.assertEqual(self.bench_cli("--rounds", "4").returncode, 2)
        self.assertEqual(self.bench_cli("--rounds", "0").returncode, 2)

    def test_modules_import_without_torch(self):
        code = ("from utils import stats, benchmark; import sys; "
                "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


@unittest.skipUnless(HAS_GPU and HAS_TRITON,
                     "requires a CUDA GPU; benchmark samples never ran here")
class GPUBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)

    def test_bench_samples_order_and_flatness(self):
        task = make_task(self.tmp.name, "triton")
        cfg = dataclasses.replace(RolloutConfig.preset("grpo"), benchmark=True)
        result = run_task(str(task), cfg)
        self.assertTrue(result["bench/available"], result)
        for impl in ("eager", "compile", "generated"):
            self.assertEqual(result[f"bench/{impl}/event_us_n"], 6)
            for kind in ("event", "wall"):
                median = result[f"bench/{impl}/{kind}_us_median"]
                self.assertTrue(math.isfinite(median) and median > 0)
        groups = [g.split(",") for g in result["bench/order"].split("|")]
        self.assertEqual(len(groups), 6)
        for position in range(3):
            self.assertEqual(sorted(g[position] for g in groups),
                             ["compile", "compile", "eager", "eager",
                              "generated", "generated"])
        self.assertNotIn("bench/generated/graph_capturable", result)
        assert_flat(self, result)

    def test_cuda_graphs_capture_replay_and_item_hack(self):
        task = make_task(self.tmp.name, "triton")
        cfg = dataclasses.replace(RolloutConfig.preset("grpo"), benchmark=True,
                                  cuda_graphs=True)
        result = run_task(str(task), cfg)
        for impl in ("eager", "compile", "generated"):
            self.assertTrue(result[f"bench/{impl}/graph_capturable"], result)
            self.assertTrue(result[f"bench/{impl}/graph_replay_ok"], result)
        self.assertTrue(result["bench/graph_all_capturable"])
        assert_flat(self, result)
        hack = make_task(self.tmp.name, "reward-hack-torchop")
        source = (hack / "model_new.py").read_text()
        (hack / "model_new.py").write_text(source.replace(
            "return self.alpha * a + b",
            "out = self.alpha * a + b\n"
            "        return out + out.sum().item() * 0"))
        result = run_task(str(hack), cfg)
        self.assertFalse(result["bench/generated/graph_capturable"], result)
        self.assertTrue(result["bench/generated/graph_error"])
        assert_flat(self, result)


if __name__ == "__main__":
    unittest.main()
