"""CUDA-graph trace tooling, checked without a GPU or Torch.

The Nsight Systems export below is SYNTHETIC: a hand-built SQLite file with the
tables and columns the renderer reads, never a GPU measurement.
"""

import json
import sqlite3
import unittest
from pathlib import Path

from scripts import render_nsys_timeline as render
from scripts.graph_trace import nsys_command
from tests.support import scratch
from utils import graph_trace_worker as worker


def synthetic_export(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, eventType INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER,
                                                 graphNodeId INTEGER, shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, correlationId INTEGER,
                                                 graphNodeId INTEGER);
    """)
    names = {1: "cudaLaunchKernel_v7000", 2: "cudaStreamBeginCapture_v10000", 3: "cudaStreamEndCapture_v10000",
             4: "cudaGraphInstantiateWithFlags_v11040", 5: "cudaMemcpyAsync_v3020", 6: "cudaGraphLaunch_v10000",
             7: "cudaStreamSynchronize_v3020", 8: "lora_shrink_kernel", 9: "lora_expand_kernel"}
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", names.items())
    # Phase labels come from worker.range_name, as on the GPU host; the third uses textId.
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, ?, ?)", [
        (1_000, 9_000, worker.range_name("generated", "forward (no graph)"), None),
        (10_000, 20_000, worker.range_name("generated", "capture"), None),
        (21_000, 30_000, None, 99)])
    db.execute("INSERT INTO StringIds VALUES (99, ?)", (worker.range_name("generated", "replay 1"),))
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)", [
        (1_100, 1_400, 1, 1), (1_500, 8_000, 2, 7),                      # forward: launch, sync
        (10_100, 10_300, 3, 2), (10_400, 10_600, 4, 1), (10_700, 10_900, 5, 1),
        (11_000, 11_200, 6, 3), (11_300, 15_000, 7, 4),                  # capture: begin, 2 launches, end, instantiate
        (21_100, 21_300, 8, 5), (21_400, 21_700, 9, 6), (21_800, 29_000, 10, 7)])  # replay: copy, graph launch, sync
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)", [
        (2_000, 3_000, 1, None, 8),                                       # direct launch in the forward
        (22_000, 23_000, 9, 101, 8), (23_100, 24_000, 9, 102, 9)])        # two graph nodes in the replay
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, NULL)", (21_500, 21_600, 8))
    db.commit()
    db.close()


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sqlite = self.root / "trace.sqlite"
        synthetic_export(self.sqlite)

    def test_phases_assign_host_calls_and_gpu_work_by_correlation(self):
        summary = render.summarize(render.assign(render.load(self.sqlite)))
        forward, capture, replay = summary["generated"]
        self.assertEqual([p["phase"] for p in summary["generated"]], ["forward (no graph)", "capture", "replay 1"])
        self.assertEqual((forward["kernels"], forward["kernels_as_graph_nodes"], forward["graph_launches"]), (1, 0, 0))
        # Capture records launches but runs nothing on the GPU.
        self.assertEqual(capture["kernels"], 0)
        self.assertEqual(capture["api_by_category"], {"capture": 3, "launch": 2})
        self.assertEqual((replay["graph_launches"], replay["kernels"], replay["kernels_as_graph_nodes"]), (1, 2, 2))
        self.assertEqual((replay["copies"], replay["copies_as_graph_nodes"]), (1, 0))
        self.assertAlmostEqual(replay["gpu_span_us"], 2.5)
        self.assertEqual(replay["distinct_kernels"], ["lora_expand_kernel", "lora_shrink_kernel"])

    def test_render_writes_svg_and_summary(self):
        summary, files = render.render(self.sqlite, self.root / "out", "SYNTHETIC")
        self.assertEqual(files, {"generated": "timeline-generated.svg"})
        text = (self.root / "out" / "timeline-generated.svg").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("<svg"))
        self.assertIn("SYNTHETIC · generated", text)
        self.assertIn("2 kernels (2 as graph nodes)", text)
        self.assertIn("1 cudaGraphLaunch", text)
        saved = json.loads((self.root / "out" / "timeline-summary.json").read_text())
        self.assertEqual(saved["phases"], summary)

    def test_api_categories(self):
        for name, category in (("cudaGraphLaunch_v10000", "graph_launch"), ("cuGraphLaunch", "graph_launch"),
                               ("cudaStreamBeginCapture_v10000", "capture"), ("cudaLaunchKernel_v7000", "launch"),
                               ("cudaMemsetAsync_v3020", "copy"), ("cudaEventRecord", "sync"), ("cudaMalloc", "other")):
            with self.subTest(name=name):
                self.assertEqual(render.api_category(name), category)

    def test_missing_tables_give_empty_trace(self):
        empty = self.root / "empty.sqlite"
        sqlite3.connect(empty).close()
        self.assertEqual(render.load(empty), {"phases": [], "apis": [], "gpu": []})


class CommandTests(unittest.TestCase):
    def test_nsys_traces_graph_nodes_and_exports_sqlite(self):
        argv = nsys_command("nsys", Path("/out/trace"), "print(1)")
        for flag in ("--trace=cuda,nvtx", "--cuda-graph-trace=node", "--export=sqlite"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[-2:], ["-c", "print(1)"])

    def test_worker_arguments(self):
        args = worker.parse_args(["--output", "w.json"])
        self.assertEqual((args.replays, args.impls), (5, ["generated", "eager"]))
        for bad in (["--replays", "0"], ["--impls", "compile"], ["--eager-forwards", "-1"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                worker.parse_args(["--output", "w.json", *bad])
        self.assertRegex(worker.range_name("eager", "replay 3"), render.PHASE)


if __name__ == "__main__":
    unittest.main()
