"""Read-only results tools over run evidence (no Modal, MCP or network)."""

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

from cloud import results_tools as tools
from tests.support import ROOT, scratch

RUNS = ROOT / "artifacts" / "runs"
FINAL = "task-stage5-gpu-acceptance-20261110T002043Z"
CSV = ROOT / "assets" / "stage5" / "stage5-largest-honest.csv"


def synthetic_run(root, run_id="task-synthetic-1", rows=None):
    """SYNTHETIC run folder; never GPU evidence."""
    path = Path(root) / run_id
    (path / "suite").mkdir(parents=True)
    rows = rows if rows is not None else [
        {"suite/task": "t", "suite/fixture": f, "suite/backend": "triton", "suite/case": "c",
         "suite/status": "passed", "reward/reason": "correct", "env/t_gen_us": 2.0,
         "env/t_eager_us": 3.0, "env/t_compile_us": float("nan"), "log/stderr_tail": "secret path",
         "artifact/report": "/root/x", "bench/device_uuid": "GPU-123", "other": "dropped"}
        for f in ("honest", "hack", "wrong", "chain_replay")]
    (path / "suite" / "suite.json").write_text(json.dumps({"rows": rows}))
    (path / "summary.json").write_text(json.dumps({"status": "pass", "counts": {"passed": len(rows)},
                                                   "source_mismatches": [], "execution": {"wall_s": 1.5}}))
    (path / "request.json").write_text(json.dumps({"run_id": run_id, "tasks": ["t"], "case_set": "smoke"}))
    return path


class RealEvidenceTests(unittest.TestCase):
    def test_honest_timings_reproduce_the_dashboard_csv(self):
        with open(CSV, newline="", encoding="utf-8") as handle:
            expected = list(csv.DictReader(handle))
        got = tools.get_honest_timings(RUNS, FINAL)
        self.assertEqual([row["id"] for row in got], [row["id"] for row in expected])
        for row, want in zip(got, expected):
            for column in tools.TIMING_COLUMNS:
                if column.endswith("_us"):
                    self.assertAlmostEqual(row[column], float(want[column]), places=3, msg=(row["id"], column))
                else:
                    self.assertEqual(row[column], want[column], (row["id"], column))

    def test_summary_and_listing(self):
        summary = tools.get_suite_summary(RUNS, FINAL)[0]
        self.assertEqual((summary["status"], summary["passed"], summary["failed"]), ("pass", 71, 0))
        self.assertEqual(summary["git_revision"][:7], "e48c177")
        listed = {row["run_id"]: row for row in tools.list_suite_runs(RUNS, prefix="task-stage5")}
        self.assertIn(FINAL, listed)
        self.assertEqual(listed["task-stage5-gpu-acceptance-20261109T222837Z"]["failed"], 19)
        # A stopped run without summary.json is not a finished run.
        self.assertNotIn("task-stage5-gpu-acceptance-20261109T190928Z", listed)

    def test_rows_page_and_never_leak_logs(self):
        page = tools.get_suite_rows(RUNS, FINAL, limit=50)
        self.assertEqual((page["total"], len(page["rows"]), page["next_cursor"]), (71, 50, 50))
        rest = tools.get_suite_rows(RUNS, FINAL, limit=50, cursor=50)
        self.assertEqual((len(rest["rows"]), rest["next_cursor"]), (21, None))
        for row in page["rows"] + rest["rows"]:
            self.assertFalse(any(k.startswith(("log/", "artifact/")) for k in row))
            self.assertNotIn("bench/device_uuid", row)
        honest = tools.get_suite_rows(RUNS, FINAL, fixtures=["honest"], tasks=["eagle_tree_step"])
        self.assertEqual(honest["total"], 2)
        json.dumps(page, allow_nan=False)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "volume"
        self.root.mkdir()
        synthetic_run(self.root)

    def test_bad_ids_and_traversal_are_refused(self):
        outside = Path(self.tmp.name) / "outside"
        synthetic_run(Path(self.tmp.name), "outside")
        for bad in ("", "../outside", "/etc/passwd", "a/b", "a\\b", ".", "..", "x" * 129, None, 7,
                    "task-missing"):
            with self.subTest(run_id=bad), self.assertRaises(tools.ResultsError):
                tools.get_suite_summary(self.root, bad)
        self.assertTrue(outside.is_dir())

    def test_arguments_are_validated(self):
        for kwargs in ({"limit": 0}, {"limit": 501}, {"limit": True}, {"cursor": -1},
                       {"fixtures": "honest"}, {"tasks": [1]}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(tools.ResultsError):
                tools.get_suite_rows(self.root, "task-synthetic-1", **kwargs)
        with self.assertRaises(tools.ResultsError):
            tools.list_suite_runs(self.root, prefix="../")

    def test_outputs_are_flat_finite_and_filtered(self):
        rows = tools.get_suite_rows(self.root, "task-synthetic-1")["rows"]
        self.assertEqual(set(rows[0]), {"suite/task", "suite/fixture", "suite/backend", "suite/case",
                                        "suite/status", "reward/reason", "env/t_gen_us",
                                        "env/t_eager_us", "env/t_compile_us"})
        self.assertIsNone(rows[0]["env/t_compile_us"])
        timings = tools.get_honest_timings(self.root, "task-synthetic-1")
        self.assertEqual([t["id"] for t in timings], ["t Triton", "t Triton chain replay"])
        self.assertIsNone(timings[0]["compile_us"])
        json.dumps(timings, allow_nan=False)

    def test_read_only_and_dependency_free(self):
        before = sorted((p, p.stat().st_mtime_ns) for p in self.root.rglob("*"))
        for name in tools.TOOLS:
            if name == "list_suite_runs":
                tools.TOOLS[name](self.root)
            elif name == "get_suite_rows":
                tools.TOOLS[name](self.root, "task-synthetic-1")
            else:
                tools.TOOLS[name](self.root, "task-synthetic-1")
        self.assertEqual(before, sorted((p, p.stat().st_mtime_ns) for p in self.root.rglob("*")))
        code = ("import sys; from cloud import results_tools; "
                "assert not {'torch', 'modal', 'mcp', 'fastapi'} & set(sys.modules)")
        subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
