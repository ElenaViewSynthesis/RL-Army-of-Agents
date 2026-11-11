"""Task-suite W&B logging with fake wandb only; no live service is contacted."""

import json
import subprocess
import sys
import types
import unittest
from pathlib import Path

from tests.support import ROOT, scratch
from tests.test_tracking import FakeWandbModule
from tracking import suite_log


def write_evidence(root, statuses=("passed", "passed")):
    """SYNTHETIC suite evidence folder; never a GPU measurement."""
    root = Path(root)
    (root / "suite").mkdir(parents=True)
    rows = [{"suite/task": "eagle_tree_step", "suite/stage": 5, "suite/backend": "triton",
             "suite/fixture": f, "suite/status": s, "reward/value": 1.0,
             "log/stderr_tail": "noise", "artifact/report": "/p"}
            for f, s in zip(("honest", "hack"), statuses)]
    counts = {k: sum(r["suite/status"] == k for r in rows) for k in ("passed", "failed", "unavailable")}
    (root / "suite" / "suite.json").write_text(json.dumps({"rows": rows, "counts": counts}))
    (root / "summary.json").write_text(json.dumps({
        "status": "pass" if not counts["failed"] else "fail", "counts": counts,
        "source_mismatches": [], "execution": {"wall_s": 12.5}}))
    (root / "request.json").write_text(json.dumps({
        "run_id": "task-synthetic-1", "tasks": ["eagle_tree_step"], "case_set": "smoke",
        "backends": "triton", "fixtures": "all", "gpu_requested": "H200"}))
    (root / "source-manifest.json").write_text(json.dumps({"git_revision": "abc1234",
                                                           "files": {"a.py": "0" * 64}}))
    (root / "suite" / "case.bin").write_bytes(b"not evidence")
    return root


class SuiteLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.folder = write_evidence(Path(self.tmp.name) / "run")

    def test_payload_drops_logs_and_keeps_provenance(self):
        payload = suite_log.suite_payload(self.folder)
        self.assertEqual(payload["wandb_id"], "h200-task-synthetic-1")
        self.assertIn("stage5", payload["tags"])
        self.assertTrue(all(not k.startswith(("log/", "artifact/")) for r in payload["rows"] for k in r))
        self.assertEqual(payload["config"]["source/git_revision"], "abc1234")
        self.assertRegex(payload["config"]["source/manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(payload["summary"]["suite/passed"], 2)
        self.assertFalse(any(f.endswith(".bin") for f in payload["files"]))
        json.dumps(payload, allow_nan=False)

    def test_inconsistent_counts_are_rejected(self):
        suite = json.loads((self.folder / "suite" / "suite.json").read_text())
        suite["counts"]["passed"] = 5
        (self.folder / "suite" / "suite.json").write_text(json.dumps(suite))
        with self.assertRaises(ValueError):
            suite_log.suite_payload(self.folder)

    def test_log_and_readback_with_fakes(self):
        payload = suite_log.suite_payload(self.folder)
        wandb = FakeWandbModule()
        self.assertEqual(suite_log.log_suite(payload, "ent", "proj", wandb), "https://fake/run")
        self.assertEqual(wandb.init_kwargs["id"], "h200-task-synthetic-1")
        logged, _ = wandb.run.logged[0]
        self.assertEqual(logged["iteration"], 0)
        self.assertEqual(logged["suite_rows"][0], "table")
        self.assertEqual(wandb.run.summary["suite/status"], "pass")
        self.assertEqual(wandb.run.finished, 0)
        self.assertEqual(len(wandb.run.artifacts[0].files), len(payload["files"]))
        fake_run = types.SimpleNamespace(summary=dict(wandb.run.summary), config=payload["config"])
        api = types.SimpleNamespace(run=lambda path: fake_run)
        self.assertEqual(suite_log.readback(payload, "ent", "proj", api), [])
        fake_run.summary["suite/passed"] = 1
        waits = []
        self.assertEqual(suite_log.readback(payload, "ent", "proj", api, attempts=3, sleep=waits.append),
                         ["suite/passed"])
        self.assertEqual(len(waits), 2)

    def test_readback_waits_for_a_lagging_summary(self):
        # The live API served an empty summary just after finish (2026-11-10).
        payload = suite_log.suite_payload(self.folder)
        full = {**payload["summary"]}
        states = [{}, {}, full]
        api = types.SimpleNamespace(run=lambda path: types.SimpleNamespace(
            summary=states.pop(0) if len(states) > 1 else states[0], config=payload["config"]))
        waits = []
        self.assertEqual(suite_log.readback(payload, "ent", "proj", api, attempts=5, sleep=waits.append), [])
        self.assertEqual(len(waits), 2)

    def test_failed_suite_finishes_nonzero(self):
        folder = write_evidence(Path(self.tmp.name) / "failed", ("passed", "failed"))
        wandb = FakeWandbModule()
        suite_log.log_suite(suite_log.suite_payload(folder), "ent", "proj", wandb)
        self.assertEqual(wandb.run.finished, 1)

    def test_dry_run_needs_no_sdk(self):
        code = ("import sys; from tracking import suite_log; "
                f"suite_log.main([{str(self.folder)!r}, '--dry-run']); "
                "assert 'wandb' not in sys.modules and 'torch' not in sys.modules")
        proc = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)[0]["rows"], 2)


if __name__ == "__main__":
    unittest.main()
