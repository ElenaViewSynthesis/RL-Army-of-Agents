"""Kernel-name recorder tests; stage outputs are synthetic fixtures, not GPU data."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.record_kernel_names as record
from tests.support import scratch
from utils.backend import BACKENDS

# Synthetic profiler fixtures mirroring the --json protocol, not captured output.
PROFILE = {"timing_measured": True, "t_eager_us": 100.0, "t_compile_us": 80.0,
           "t_gen_us": 50.0, "provenance": 0.9,
           "provenance/agent": 0.8, "provenance/library": 0.0,
           "provenance/torch": 0.1, "provenance/memory": 0.1,
           "kernel_us/axpby_kernel_0d1d2d": 5.0,
           "kernel_us/at::native::vectorized_elementwise_kernel": 1.0}
CPU_NULL = {"timing_measured": False, "t_eager_us": None, "t_compile_us": None,
            "t_gen_us": None, "provenance": None, "provenance/agent": None,
            "provenance/library": None, "provenance/torch": None,
            "provenance/memory": None}


def fake_backends(**overrides):
    base = {name: "package not installed: fake" for name in BACKENDS}
    base.update(overrides)
    return base


def run_main(profile, *argv):
    stages = [(0, "Compile success\n", "", False),
              (0, "timing line\n" + json.dumps(profile), "", False)]
    with patch("utils.rollout_env._stage", side_effect=stages), patch(
            "scripts.record_kernel_names.available_backends",
            return_value=fake_backends(triton=None)):
        return record.main(list(argv))


class RecordTests(unittest.TestCase):
    def test_measured_classes_and_unmatched(self):
        with scratch() as tmp:
            out = Path(tmp) / "k.json"
            run_main(PROFILE, "--device", "cuda", "--backends", "triton",
                     "--output", str(out))
            triton = json.loads(out.read_text())["backends"]["triton"]
        self.assertTrue(triton["measured"])
        self.assertEqual(triton["agent_kernel_names"], ["axpby_kernel"])
        self.assertEqual(triton["kernels"]["axpby_kernel_0d1d2d"],
                         {"us": 5.0, "class": "agent"})
        self.assertEqual(triton["kernels"]["at::native::vectorized_elementwise_kernel"],
                         {"us": 1.0, "class": "torch"})
        self.assertEqual(triton["unmatched_agent_names"], [])

    def test_unknown_kernel_counts_against_authored(self):
        profile = dict(PROFILE, **{"kernel_us/mystery_kernel": 2.0})
        del profile["kernel_us/axpby_kernel_0d1d2d"]
        del profile["kernel_us/at::native::vectorized_elementwise_kernel"]
        with scratch() as tmp:
            out = Path(tmp) / "k.json"
            run_main(profile, "--device", "cuda", "--backends", "triton",
                     "--output", str(out))
            triton = json.loads(out.read_text())["backends"]["triton"]
        self.assertEqual(triton["kernels"]["mystery_kernel"]["class"], "torch")
        self.assertEqual(triton["unmatched_agent_names"], ["axpby_kernel"])

    def test_cpu_null_record_is_unmeasured(self):
        with scratch() as tmp:
            out = Path(tmp) / "k.json"
            run_main(CPU_NULL, "--device", "cpu", "--backends", "triton",
                     "--output", str(out))
            triton = json.loads(out.read_text())["backends"]["triton"]
        self.assertEqual(triton, {"measured": False, "reason": "CPU (no device events)"})

    def test_unavailable_backend_skips_stages(self):
        with scratch() as tmp:
            out = Path(tmp) / "k.json"
            with patch("utils.rollout_env._stage") as stage, patch(
                    "scripts.record_kernel_names.available_backends",
                    return_value=fake_backends()):
                record.main(["--device", "cuda", "--backends", "cuda",
                             "--output", str(out)])
            stage.assert_not_called()
            cuda = json.loads(out.read_text())["backends"]["cuda"]
        self.assertEqual(cuda, {"unavailable": "package not installed: fake"})


if __name__ == "__main__":
    unittest.main()
