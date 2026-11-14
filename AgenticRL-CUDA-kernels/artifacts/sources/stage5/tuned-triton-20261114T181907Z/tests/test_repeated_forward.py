"""CPU interpreter regressions for repeated, immutable model inputs."""
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tests.support import ROOT


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('torch','triton','numpy')),
                     'Torch, Triton and NumPy required for repeated CPU forwards')
class RepeatedForwards(unittest.TestCase):
    def test_stage6_and_stage5_shared_inputs_for_40_calls(self):
        process=subprocess.run([sys.executable,'-B',str(ROOT/'tests/repeated_forward_probe.py')],cwd=ROOT,
            capture_output=True,text=True,timeout=900,
            env={**os.environ,'CUDA_AGENT_DEVICE':'cpu','TRITON_INTERPRET':'1',
                 'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'})
        target=os.environ.get('STAGE6_REPEAT_LOG')
        if target:Path(target).write_text(process.stdout+process.stderr)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.assertIn('REPEATED_FORWARD_PROBES_OK',process.stdout)
