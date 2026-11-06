"""Runtime preflight tests; injected fakes, never real host probes."""

import json
import os
import subprocess
import sys
import unittest

from tests.support import ROOT, assert_flat
from utils import runtime

# Layout fixture mirroring the documented `--format=csv,noheader,nounits`
# output; not a captured host output.
HOPPER_SMI = ("0, NVIDIA H200, 580.65.06, 9.0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n"
              "1, NVIDIA H200, 580.65.06, 9.0, GPU-ffffffff-0000-1111-2222-333333333333\n")
AMPERE_SMI = "0, NVIDIA A100, 580.65.06, 8.0, GPU-bbbbbbbb-1111-2222-3333-444444444444\n"
NVCC_133 = ("nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 13.3, V13.3.52\n")


def fake_env(smi=None, nvcc=None, which_paths=(), packages=None):
    def run(cmd, timeout_s=20):
        return {"nvidia-smi": smi, "nvcc": nvcc}.get(cmd[0])

    def which(tool):
        return f"/usr/bin/{tool}" if tool in which_paths else None

    def pkg(dist):
        return (packages or {}).get(dist)

    return dict(run=run, which=which, pkg=pkg)


class ParseTests(unittest.TestCase):
    def test_version_tuple(self):
        self.assertEqual(runtime.version_tuple("580.65.06"), (580, 65, 6))
        self.assertEqual(runtime.version_tuple("13.3"), (13, 3))
        self.assertEqual(runtime.version_tuple("2.14.1+cu130"), (2, 14, 1))
        self.assertEqual(runtime.version_tuple(None), ())
        self.assertEqual(runtime.version_tuple("on PATH"), ())

    def test_parse_nvidia_smi(self):
        gpus = runtime.parse_nvidia_smi(HOPPER_SMI + "\ngarbage line\n")
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0], {"index": 0, "name": "NVIDIA H200",
                                   "driver_version": "580.65.06", "compute_cap": "9.0",
                                   "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"})
        self.assertEqual(runtime.parse_nvidia_smi(""), [])

    def test_parse_nvcc_version(self):
        self.assertEqual(runtime.parse_nvcc_version(NVCC_133), "13.3")
        self.assertIsNone(runtime.parse_nvcc_version("garbage"))


class CheckTests(unittest.TestCase):
    def test_everything_missing(self):
        record = runtime.check(**fake_env())
        self.assertEqual(record["runtime/gpu_count"], 0)
        self.assertFalse(record["runtime/ok_cuda_backend"])
        self.assertFalse(record["runtime/ok_triton_backend"])
        self.assertFalse(record["runtime/ok_cutile"])
        self.assertIn("nvidia-smi", record["runtime/unavailable_reason"])
        assert_flat(self, record)

    def test_hopper_needs_tileiras_133(self):
        env = fake_env(smi=HOPPER_SMI, nvcc=NVCC_133,
                       packages={"nvidia-cuda-tileiras": "13.2.10"})
        record = runtime.check(**env)
        self.assertFalse(record["runtime/tileiras_ok"])
        self.assertIn("13.3", record["runtime/tileiras_reason"])
        env["pkg"] = fake_env(packages={"nvidia-cuda-tileiras": "13.3.52"})["pkg"]
        record = runtime.check(**env)
        self.assertTrue(record["runtime/tileiras_ok"])

    def test_ampere_accepts_tileiras_131(self):
        env = fake_env(smi=AMPERE_SMI, packages={"nvidia-cuda-tileiras": "13.1.0"})
        record = runtime.check(**env)
        self.assertTrue(record["runtime/tileiras_ok"])

    def test_happy_path_all_profiles(self):
        env = fake_env(smi=HOPPER_SMI, nvcc=NVCC_133, which_paths=("ninja",),
                       packages={"torch": "2.14.1+cu130", "triton": "3.6.0",
                                 "cuda-tile": "1.8.0", "nvidia-cuda-tileiras": "13.3.52"})
        record = runtime.check(**env)
        self.assertTrue(record["runtime/ok_cuda_backend"])
        self.assertTrue(record["runtime/ok_triton_backend"])
        self.assertTrue(record["runtime/ok_cutile"])
        self.assertIsNone(record["runtime/unavailable_reason"])
        self.assertTrue(record["runtime/torch_cuda_build"])
        assert_flat(self, record)


class CLITests(unittest.TestCase):
    def test_json_line_and_require_exit_codes(self):
        proc = subprocess.run([sys.executable, "-S", "-m", "utils.runtime", "--json"],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        record = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertFalse(record["runtime/ok_cuda_backend"])
        assert_flat(self, record)
        proc = subprocess.run([sys.executable, "-S", "-m", "utils.runtime",
                               "--require", "cutile"], cwd=ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)

    def test_import_without_torch(self):
        code = "from utils import runtime; import sys; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
