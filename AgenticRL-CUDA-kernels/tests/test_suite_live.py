"""Live task-suite W&B progress with a fake volume and fake wandb only; no service is contacted."""

import io
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path

from tests.support import ROOT, scratch
from tests.test_tracking import FakeWandbModule
from tracking import contract, suite_live


def row(task, fixture, status, reward, gen=None):
    """SYNTHETIC suite row; never a GPU measurement."""
    return {"suite/task": task, "suite/case": "c1", "suite/backend": "triton", "suite/fixture": fixture,
            "suite/status": status, "reward/value": reward, "env/t_gen_us": gen,
            "env/t_eager_us": 2.0, "env/t_compile_us": float("nan"), "env/provenance_agent": 1.0,
            "suite/wall_s": 3.5, "log/stderr_tail": "noise"}


class FakeVolume:
    """Serves a scripted sequence of volume states, one per read of progress.jsonl."""

    def __init__(self, request=None):
        self.files = {"request.json": json.dumps(request or {"run_id": "r1", "case_set": "smoke",
                                                             "backends": "triton", "fixtures": "all",
                                                             "tasks": ["t"], "gpu_requested": "H200"})}

    def read(self, relative):
        return self.files.get(relative)

    def set_rows(self, rows, partial_tail=""):
        self.files["suite/progress.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows) + partial_tail


class PureHelperTests(unittest.TestCase):
    def test_parse_progress_stops_at_a_partial_last_line(self):
        text = json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n" + '{"a": 3'
        self.assertEqual(suite_live.parse_progress(text), [{"a": 1}, {"a": 2}])
        self.assertEqual(suite_live.parse_progress(""), [])

    def test_row_metrics_keep_finite_numbers_only(self):
        counts = {"passed": 1, "failed": 0, "unavailable": 0}
        metrics = suite_live.row_metrics(1, row("t", "honest", "passed", 2.0, gen=1.5), counts)
        self.assertEqual(metrics["live/row"], 1)
        self.assertEqual(metrics["live/passed"], 1)
        self.assertEqual(metrics["live/row_status_passed"], 1)
        self.assertEqual(metrics["live/row_t_gen_us"], 1.5)
        self.assertNotIn("live/row_t_compile_us", metrics)  # NaN is dropped, never logged as a value
        missing = suite_live.row_metrics(2, row("t", "wrong", "passed", -1.0, gen=None), counts)
        self.assertNotIn("live/row_t_gen_us", missing)
        self.assertTrue(all(not k.startswith(("log/", "env/", "suite/")) for k in metrics))
        json.dumps(metrics, allow_nan=False)

    def test_every_live_metric_has_one_adr_classification(self):
        counts = {"passed": 0, "failed": 0, "unavailable": 0}
        for key in suite_live.row_metrics(1, row("t", "honest", "passed", 2.0, gen=1.0), counts):
            self.assertEqual(contract.classify("suite_live", key), "metric")


class LiveSuiteTests(unittest.TestCase):
    def test_rows_are_logged_once_in_order_and_run_finishes(self):
        volume, wandb, out = FakeVolume(), FakeWandbModule(), io.StringIO()
        live = suite_live.LiveSuite("r1", volume, "ent", "proj", wandb_module=wandb, out=out)
        self.assertEqual(live.poll(), 0)  # no progress file yet, but the run is started
        self.assertEqual(wandb.init_kwargs["id"], "h200-r1")
        self.assertEqual(wandb.init_kwargs["resume"], "allow")
        self.assertIn("live", wandb.init_kwargs["tags"])
        self.assertIn(("live/*", {"step_metric": "live/row", "summary": "none"}), wandb.run.definitions)
        volume.set_rows([row("t", "honest", "passed", 2.0, 1.0)], partial_tail='{"suite/ta')
        self.assertEqual(live.poll(), 1)
        volume.set_rows([row("t", "honest", "passed", 2.0, 1.0), row("t", "hack", "failed", -1.0, 3.0)])
        self.assertEqual(live.poll(), 1)
        self.assertEqual(live.poll(), 0)
        logged = [entry for entry, _ in wandb.run.logged]
        self.assertEqual([m["live/row"] for m in logged], [1, 2])
        self.assertEqual((logged[1]["live/passed"], logged[1]["live/failed"]), (1, 1))
        self.assertEqual(wandb.run.summary, {})  # no last-row summaries
        volume.files["summary.json"] = json.dumps({"status": "fail"})
        summary = suite_live.watch(live, interval_s=0, sleep=lambda _: None)
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(wandb.run.finished, 1)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(lines[-1], {"event": "finished", "status": "fail", "rows": 2,
                                     "passed": 1, "failed": 1, "unavailable": 0})

    def test_waits_for_the_request_before_starting(self):
        volume, wandb = FakeVolume(), FakeWandbModule()
        del volume.files["request.json"]
        live = suite_live.LiveSuite("r1", volume, "ent", "proj", wandb_module=wandb, out=io.StringIO())
        self.assertEqual(live.poll(), 0)
        self.assertIsNone(wandb.init_kwargs)

    def test_watch_gives_up_after_max_wait(self):
        volume, wandb = FakeVolume(), FakeWandbModule()
        live = suite_live.LiveSuite("r1", volume, "ent", "proj", wandb_module=wandb, out=io.StringIO())
        ticks = iter([0.0, 10.0, 20.0])
        waits = []
        self.assertIsNone(suite_live.watch(live, interval_s=5, max_wait_s=15, sleep=waits.append,
                                           clock=lambda: next(ticks)))
        self.assertEqual(waits, [5])
        self.assertEqual(wandb.run.finished, 1)

    def test_dry_run_prints_and_never_imports_the_sdk(self):
        files = {"request.json": json.dumps({"run_id": "r1", "case_set": "smoke"}),
                 "suite/progress.jsonl": json.dumps({"suite/status": "passed", "reward/value": 2.0}) + "\n",
                 "summary.json": json.dumps({"status": "pass"})}
        code = ("import io, json, sys, types; from tracking import suite_live; "
                f"files = json.loads({json.dumps(files)!r}); "
                "v = types.SimpleNamespace(read=files.get); out = io.StringIO(); "
                "live = suite_live.LiveSuite('r1', v, dry_run=True, out=out); "
                "suite_live.watch(live, interval_s=0, sleep=lambda _: None); "
                "assert 'wandb' not in sys.modules and 'torch' not in sys.modules; print(out.getvalue())")
        proc = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[0]["wandb_id"], "h200-r1")
        self.assertEqual(events[-1]["event"], "finished")


class ModalVolumeTests(unittest.TestCase):
    def test_read_uses_the_cli_and_returns_none_when_missing(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            if argv[-2].endswith("request.json"):
                Path(argv[-1]).write_text('{"run_id": "r1"}')
                return types.SimpleNamespace(returncode=0, stderr="")
            return types.SimpleNamespace(returncode=1, stderr="No such file")

        volume = suite_live.ModalVolume("r1", modal="/bin/modal", runner=runner)
        self.assertEqual(json.loads(volume.read("request.json")), {"run_id": "r1"})
        self.assertIsNone(volume.read("summary.json"))
        self.assertEqual(calls[0][:5], ["/bin/modal", "volume", "get", "--force", suite_live.VOLUME])
        self.assertEqual(calls[0][5], "/r1/request.json")


class CollectEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_copies_the_evidence_subset_only(self):
        run = self.root / "dl" / "r1"
        for relative, text in {"summary.json": "{}", "suite/progress.jsonl": "", "suite.stdout.log": "x",
                               "suite/t/c/triton/honest/README.md": "#", "suite/t/c/triton/honest/case.bin": "b",
                               "suite/t/c/triton/honest/.artifacts/p.json": "{}",
                               "suite/t/c/triton/honest/build/x.json": "{}",
                               "suite/t/c/triton/honest/source/model.py": "",
                               "source/task-source-manifest.json": "{}", "source/utils/a.json": "{}"}.items():
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        client = self.root / "client.log"
        client.write_text("modal")
        dest = suite_live.collect_evidence(run, self.root / "runs" / "r1", client)
        got = sorted(str(p.relative_to(dest)).replace("\\", "/") for p in dest.rglob("*") if p.is_file())
        self.assertEqual(got, ["modal-session.log", "source/task-source-manifest.json", "suite.stdout.log",
                               "suite/progress.jsonl", "suite/t/c/triton/honest/README.md", "summary.json"])
        with self.assertRaises(FileExistsError):
            suite_live.collect_evidence(run, dest)


if __name__ == "__main__":
    unittest.main()
