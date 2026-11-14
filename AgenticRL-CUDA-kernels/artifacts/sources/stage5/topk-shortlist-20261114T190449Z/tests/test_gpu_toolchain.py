"""Guard the cloud bindings pin against the observed H200 driver API ceiling."""
import re
import unittest
from pathlib import Path

from tests.support import ROOT


class GPUDriverBindingsTests(unittest.TestCase):
    def test_bindings_api_does_not_exceed_captured_driver(self):
        requirements = (ROOT / 'requirements-gpu.txt').read_text()
        pins = re.findall(r'^cuda-bindings==(\d+)\.(\d+)\.(\d+)\s*$', requirements, re.M)
        self.assertEqual(len(pins), 1, 'Pin CUDA bindings explicitly; a floating dependency regressed memcheck')
        major, minor, _ = map(int, pins[0])
        log = (ROOT / 'tests/fixtures/diagnostics/16-sanitizer-cutedsl-memcheck/sanitizer.log').read_text()
        failures = re.findall(r'cudaVersion argument \((\d+)\) exceeds the driver version \((\d+)\)', log)
        self.assertTrue(failures, 'Retain the original API mismatch as evidence')
        ceilings = {int(driver) for _, driver in failures}
        self.assertEqual(len(ceilings), 1)
        self.assertLessEqual(major * 1000 + minor * 10, ceilings.pop())
        self.assertTrue(all(int(requested) > int(driver) for requested, driver in failures))


if __name__ == '__main__':
    unittest.main()
