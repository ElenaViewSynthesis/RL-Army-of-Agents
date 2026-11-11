import argparse
import dataclasses
import json
import subprocess
import sys
import unittest

from tests.support import ROOT, assert_flat, mock_rollout
from utils.config import ALGOS, RolloutConfig
from utils.grpo import build_batch, filter_groups, group_advantages
from utils.reward import compute_reward


class ConfigTests(unittest.TestCase):
    def parse(self, flags):
        parser = argparse.ArgumentParser()
        RolloutConfig.add_args(parser)
        return RolloutConfig.from_args(parser.parse_args(flags))

    def test_presets_and_independent_overrides(self):
        grpo, dapo = RolloutConfig.preset("grpo"), RolloutConfig.preset("dapo")
        self.assertEqual((grpo.clip_low, grpo.clip_high, grpo.loss_agg, grpo.kl_coef), (.2, .2, "seq", .001))
        self.assertFalse(grpo.filter_tied_groups)
        self.assertEqual(grpo.overlong_buffer_turns, 0)
        self.assertEqual((dapo.clip_low, dapo.clip_high, dapo.loss_agg, dapo.kl_coef), (.2, .28, "token", 0))
        self.assertTrue(dapo.filter_tied_groups)
        self.assertEqual(dapo.overlong_buffer_turns, 4)
        cfg = self.parse(["--clip-high", "0.3", "--algo", "dapo", "--no-filter-tied-groups",
                          "--loss-agg", "seq", "--overlong-buffer-turns", "0", "--adv-norm", "none"])
        self.assertEqual((cfg.clip_high, cfg.filter_tied_groups, cfg.loss_agg, cfg.overlong_buffer_turns),
                         (.3, False, "seq", 0))
        self.assertEqual(cfg.adv_norm, "none")
        self.assertEqual(self.parse([]).to_dict(), grpo.to_dict())

    def test_cispo_preset_and_is_weight_bounds(self):
        cispo = RolloutConfig.preset("cispo")
        # IS weight clipped to [1 - 1.0, 1 + 4.0] = [0, 5]: no lower bound.
        self.assertEqual((cispo.clip_low, cispo.clip_high, cispo.loss_agg, cispo.kl_coef), (1, 4, "token", 0))
        self.assertEqual((cispo.adv_norm, cispo.filter_tied_groups, cispo.overlong_buffer_turns), ("std", True, 4))
        self.assertEqual(self.parse(["--algo", "cispo"]).to_dict(), cispo.to_dict())
        self.assertEqual(self.parse(["--algo", "cispo", "--clip-high", "1.0"]).clip_high, 1)
        RolloutConfig(algo="cispo", clip_low=0.2)
        for kwargs in ({"algo": "cispo", "clip_low": 1.01}, {"algo": "cispo", "clip_low": -.1},
                       {"algo": "grpo", "clip_low": 1}, {"algo": "dapo", "clip_low": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RolloutConfig(**kwargs)

    def test_every_field_has_a_cli_flag(self):
        flags = ["--backend", "triton", "--iters", "3", "--benchmark",
                 "--bench-warmup-iters", "4", "--bench-rounds", "3",
                 "--bench-iters-per-round", "2", "--cuda-graphs", "--timeout-s", "2.5",
                 "--provenance-threshold", ".95", "--speedup-margin", ".1",
                 "--max-turns", "30", "--max-context-tokens", "4096", "--algo", "dapo",
                 "--group-size", "3", "--adv-norm", "none", "--filter-tied-groups",
                 "--overlong-buffer-turns", "5", "--clip-low", ".1", "--clip-high", ".3",
                 "--loss-agg", "seq", "--kl-coef", ".01", "--seed", "17", "--run-name", "trial",
                 "--tags", "a", "b"]
        cfg = self.parse(flags)
        self.assertTrue(cfg.benchmark)
        self.assertTrue(cfg.cuda_graphs)
        self.assertEqual((cfg.bench_warmup_iters, cfg.bench_rounds,
                          cfg.bench_iters_per_round), (4, 3, 2))
        self.assertFalse(self.parse(["--no-benchmark", "--no-cuda-graphs"]).benchmark)
        self.assertEqual(cfg.timeout_s, 2.5)
        self.assertEqual(cfg.tags, ["a", "b"])
        self.assertEqual(cfg.seed, 17)
        self.assertEqual(list(cfg.to_dict()), [f.name for f in dataclasses.fields(cfg)])
        self.assertEqual(json.loads(json.dumps(cfg.to_dict(), allow_nan=False)), cfg.to_dict())
        clone = cfg.to_dict()
        clone["tags"].append("other")
        self.assertEqual(cfg.tags, ["a", "b"])

    def test_invalid_configs(self):
        for kwargs in ({"timeout_s": 0}, {"iters": 0}, {"bench_rounds": 4},
                       {"bench_rounds": 0}, {"bench_warmup_iters": 0},
                       {"benchmark": "yes"}, {"group_size": 0}, {"max_turns": 0},
                       {"max_context_tokens": 0}, {"provenance_threshold": 1.1}, {"speedup_margin": 1},
                       {"overlong_buffer_turns": 21}, {"timeout_s": float("nan")},
                       {"backend": "other"}, {"adv_norm": "other"}, {"clip_low": 1},
                       {"loss_agg": "other"}, {"kl_coef": -1}, {"tags": "not-a-list"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RolloutConfig(**kwargs)

    def test_environment_modules_import_without_site_packages(self):
        code = ("from utils import config, reward, grpo, rollout_env, stats, benchmark, runtime; "
                "import sys; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RolloutConfig.preset("grpo")

    def reward(self, eager=100, compile=80, generated=50, provenance=1, turns=None, cfg=None, ok=True):
        row = compute_reward(ok, eager, compile, generated, provenance, cfg or self.cfg, turns)
        assert_flat(self, row)
        self.assertIn(row["reward_base"], (-1, 1, 2, 3))
        return row

    def test_milestone_exact_boundaries(self):
        for generated, base in ((50, 3), (76, 2), (94, 2), (95, 1), (120, 1)):
            self.assertEqual(self.reward(generated=generated)["reward_base"], base)
        self.assertEqual(self.reward(eager=80, compile=100, generated=90)["reward_base"], 1)
        self.assertEqual(self.reward(provenance=.9)["reward_base"], 3)

    def test_gates_and_missing_measurements(self):
        self.assertEqual(self.reward(ok=False)["reason"], "correctness")
        for share in (.899, 0, float("nan"), float("inf"), 1.1):
            self.assertEqual(self.reward(provenance=share)["reason"], "provenance")
        self.assertEqual(self.reward(provenance=None)["reason"], "unmeasured")
        for timing in (0, -1, None, float("nan"), float("inf")):
            self.assertEqual(self.reward(generated=timing)["reason"], "timing")
        self.assertEqual(self.reward(provenance={"agent": .85, "library": .05})["reward_base"], 3)

    def test_turn_caps_and_dapo_shaping(self):
        dapo = RolloutConfig.preset("dapo")
        self.assertEqual(self.reward(turns=19)["overlong_penalty"], 0)
        for turns, penalty in ((16, 0), (17, .25), (19, .75), (20, 1)):
            result = self.reward(cfg=dapo, turns=turns)
            self.assertEqual(result["overlong_penalty"], penalty)
            self.assertEqual(result["reward_base"], 3)
            self.assertEqual(result["reward"], 3 - penalty)
        for cfg in (self.cfg, dapo):
            result = self.reward(cfg=cfg, turns=21)
            self.assertEqual((result["reward"], result["reason"], result["overlong_penalty"]), (-1, "max_turns", 0))
        self.assertEqual(self.reward(cfg=dapo, turns=None)["reward"], 3)
        self.assertEqual(self.reward(cfg=dapo, turns=19, ok=False)["overlong_penalty"], 0)
        self.assertEqual(self.reward(cfg=dapo, turns=19, provenance=0)["reward"], -1)
        with self.assertRaises(ValueError):
            self.reward(turns=-1)

    def test_configurable_margin(self):
        cfg = dataclasses.replace(self.cfg, speedup_margin=.1)
        self.assertEqual(self.reward(generated=90, cfg=cfg)["reward_base"], 1)


class GroupTests(unittest.TestCase):
    def test_identical_groups_under_every_preset(self):
        rows = ([mock_rollout(1) for _ in range(3)] +
                [mock_rollout(3), mock_rollout(1), mock_rollout(-1, "provenance")])
        for algo in ALGOS:
            cfg = dataclasses.replace(RolloutConfig.preset(algo), group_size=3)
            pairs, stats = build_batch(["tied"] * 3 + ["mixed"] * 3, rows, cfg)
            assert_flat(self, stats)
            advantages = dict(pairs)
            if not cfg.filter_tied_groups:  # GRPO keeps tied groups; DAPO and CISPO drop them
                self.assertEqual([advantages[str(i)] for i in range(3)], [0, 0, 0])
                self.assertEqual(stats["adv/frac_dropped"], 0)
            else:
                self.assertEqual(len(pairs), 3)
                self.assertEqual(stats["adv/frac_dropped"], .5)
            self.assertAlmostEqual(sum(advantages[str(i)] for i in range(3, 6)), 0)
            self.assertLess(advantages["5"], advantages["4"])
            self.assertLess(advantages["5"], advantages["3"])
            self.assertAlmostEqual(stats["reward/frac_provenance_fail"], 1 / 6)

    def test_normalization_filter_and_empty(self):
        self.assertEqual(group_advantages([3, 1, -1], "none"), [2, 0, -2])
        self.assertEqual(group_advantages([1] * 8, "std"), [0] * 8)
        for norm in ("std", "none"):
            self.assertAlmostEqual(sum(group_advantages([-1, 1, 2, 3], norm)), 0)
        self.assertEqual(filter_groups([[1, 1], [-1, 3]], True), ([[-1, 3]], 1))
        self.assertEqual(filter_groups([[1, 1], [-1, 3]], False), ([[1, 1], [-1, 3]], 0))
        pairs, stats = build_batch([], [], RolloutConfig.preset("dapo"))
        self.assertEqual(pairs, [])
        self.assertTrue(all(value == 0 for value in stats.values()))
        assert_flat(self, stats)

    def test_multiple_groups_same_task_and_stats(self):
        cfg = dataclasses.replace(RolloutConfig.preset("dapo"), group_size=2, adv_norm="none")
        rows = [mock_rollout(3, turns=1), mock_rollout(3, turns=2),
                mock_rollout(2.25, turns=19, penalty=.75), mock_rollout(-1, "max_turns", turns=25)]
        pairs, stats = build_batch(["task"] * 4, rows, cfg)
        self.assertEqual(pairs, [("2", 1.625), ("3", -1.625)])
        self.assertEqual(stats["turns/p95"], 25)
        self.assertEqual(stats["reward/frac_max_turns"], .25)
        self.assertEqual(stats["reward/mean_overlong_penalty"], .75 / 4)
        self.assertEqual(stats["turns/mean"], 47 / 4)

    def test_reject_partial_and_invalid_groups(self):
        with self.assertRaises(ValueError):
            build_batch(["task"], [mock_rollout()], RolloutConfig())
        with self.assertRaises(ValueError):
            build_batch([], [mock_rollout()], RolloutConfig())
        with self.assertRaises(ValueError):
            group_advantages([float("nan")])
        with self.assertRaises(ValueError):
            group_advantages([1], "invalid")
        with self.assertRaises(ValueError):
            filter_groups([[]], True)


if __name__ == "__main__":
    unittest.main()
