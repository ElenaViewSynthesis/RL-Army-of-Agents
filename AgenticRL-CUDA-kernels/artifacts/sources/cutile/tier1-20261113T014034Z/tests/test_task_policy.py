import json
import unittest

from utils.config import RolloutConfig
from utils.task_policy import apply_task_policy, authored_fraction, diagnostic_rates


class TaskPolicyTests(unittest.TestCase):
    def record(self, provenance=1.0, reason="faster_than_both"):
        return {"env/verify_ok": True, "env/t_eager_us": 100.0,
                "env/t_compile_us": 80.0, "env/t_gen_us": 50.0,
                "env/provenance": provenance, "reward/value": 3.0,
                "reward/base": 3, "reward/reason": reason, "reward/overlong_penalty": 0.0}

    def profile(self, agent, library=0.0, torch=0.0, memory=0.0):
        return {"timing_measured": True, "provenance/agent": agent,
                "provenance/library": library, "provenance/torch": torch,
                "provenance/memory": memory}

    def test_synthetic_cuda_library_hack_rejected_without_redefining_env_share(self):
        original = self.record()
        result = apply_task_policy(original, self.profile(0, library=1), RolloutConfig())
        self.assertEqual(result["env/provenance"], 1.0)
        self.assertEqual(result["task/provenance_authored"], 0.0)
        self.assertEqual((result["reward/value"], result["reward/reason"]), (-1.0, "provenance"))
        self.assertEqual(result["task/policy_violation"], "library_kernel_time")
        self.assertEqual(original["reward/value"], 3.0)

    def test_memory_excluded_and_boundary_inclusive(self):
        profile = self.profile(0.45, torch=0.05, memory=0.5)
        self.assertAlmostEqual(authored_fraction(profile), 0.9)
        result = apply_task_policy(self.record(), profile, RolloutConfig())
        self.assertEqual(result["reward/value"], 3.0)
        self.assertIsNone(result["task/policy_violation"])

    def test_no_library_even_when_authored_fraction_exceeds_threshold(self):
        result = apply_task_policy(self.record(), self.profile(.99, library=.01), RolloutConfig())
        self.assertEqual(result["reward/reason"], "provenance")

    def test_task_floor_cannot_be_lowered_by_caller(self):
        result = apply_task_policy(self.record(), self.profile(.8, torch=.2),
                                   RolloutConfig(provenance_threshold=.5))
        self.assertEqual(result["reward/reason"], "provenance")
        self.assertEqual(result["task/provenance_threshold"], .9)

    def test_invalid_share_records_fail_closed(self):
        for profile in ({}, self.profile(float("nan")), self.profile(-1, torch=2),
                        self.profile(0), self.profile(True), self.profile(.5)):
            with self.subTest(profile=profile):
                self.assertIsNone(authored_fraction(profile))
                result = apply_task_policy(self.record(), profile, RolloutConfig())
                self.assertEqual(result["reward/reason"], "provenance")
                json.dumps(result, allow_nan=False)

    def test_existing_failure_gates_and_shaping_are_preserved(self):
        for reason in ("verification", "timeout", "empty_profile:eager", "unmeasured"):
            original = self.record(reason=reason)
            original.update({"reward/value": -1.0, "reward/base": -1})
            result = apply_task_policy(original, self.profile(0, library=1), RolloutConfig())
            self.assertEqual(result["reward/reason"], reason)
            self.assertEqual(result["reward/value"], -1)
        shaped = apply_task_policy(self.record(), self.profile(1),
                                   RolloutConfig.preset("dapo"), turns_used=19)
        self.assertEqual(shaped["reward/value"], 2.25)

    def test_diagnostic_profile_cannot_be_reward_evidence(self):
        profile = {**self.profile(1), "profile/diagnostic": True}
        result = apply_task_policy(self.record(), profile, RolloutConfig())
        self.assertEqual((result["reward/value"], result["reward/reason"]),
                         (-1, "diagnostic_profile"))

    def test_synthetic_diagnostic_rates_are_unit_correct(self):
        result = diagnostic_rates({"flops": 10**12, "logical_io_bytes": 10**9,
                                   "flop_convention": "synthetic",
                                   "bandwidth_convention": "synthetic"}, 1000.0)
        self.assertEqual(result["bench/generated/tflops"], 1000.0)
        self.assertEqual(result["bench/generated/logical_gbytes_per_s"], 1000.0)
        for latency in (None, 0, -1, float("inf"), float("nan")):
            result = diagnostic_rates({"flops": 1, "logical_io_bytes": 1}, latency)
            self.assertIsNone(result["bench/generated/tflops"])
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
