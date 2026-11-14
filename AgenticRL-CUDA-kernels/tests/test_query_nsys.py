"""scripts/query_nsys.py, checked without nsys, a GPU or Torch.

Two inputs: the recorded H200 fixture (one axpby launch), and a SYNTHETIC export
built below with the tables and columns the script reads. The synthetic file is
never a GPU measurement.
"""

import contextlib
import io
import json
import sqlite3
import unittest
from pathlib import Path

from scripts import query_nsys as q
from tests.support import scratch

FIXTURE = Path(__file__).parent / "fixtures" / "diagnostics" / "nsys-cuda-recovered" / "capture.sqlite"
PID = 5 << 24
MAIN, WORKER = PID + 1, PID + 2


def synthetic_export(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, name TEXT, label TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER, domainId INTEGER,
                                  globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER,
                                                  correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER, globalPid INTEGER,
                                                 shortName INTEGER, demangledName INTEGER, gridX INTEGER, gridY INTEGER,
                                                 gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                                                 registersPerThread INTEGER, graphNodeId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, correlationId INTEGER, globalPid INTEGER,
                                                 bytes INTEGER, copyKind INTEGER);
    """)
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", {
        1: "cudaLaunchKernel_v7000", 2: "cudaMemcpyAsync_v3020", 3: "shrink_kernel", 4: "expand_kernel",
        5: "void shrink_kernel(float *)", 6: "void expand_kernel(float *)", 7: "inner"}.items())
    db.executemany("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?, ?, ?)", [
        (1, "CUDA_MEMCPY_KIND_HTOD", "Host-to-Device"), (2, "CUDA_MEMCPY_KIND_DTOH", "Device-to-Host")])
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)", [
        (1_000, 9_000, "outer", None, 0, MAIN),
        (2_000, 5_000, None, 7, 0, MAIN),          # "inner", by textId, nested in "outer"
        (6_000, 8_000, None, 7, 0, MAIN),          # a second "inner"
        (1_500, 2_500, "other thread", None, 3, WORKER),
        (4_000, None, "a mark", None, 0, MAIN)])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", [
        (2_100, 2_300, MAIN, 11, 1), (2_400, 2_600, MAIN, 12, 1),   # two launches inside the first "inner"
        (6_100, 6_300, MAIN, 13, 1),                                # one launch inside the second "inner"
        (8_100, 8_400, MAIN, 14, 2),                                # a copy inside "outer" only
        (1_600, 1_800, WORKER, 15, 2)])                             # a copy on the other thread
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        (3_000, 3_400, 11, PID, 3, 5, 8, 1, 1, 256, 1, 1, 20, None),
        (3_500, 4_500, 12, PID, 4, 6, 4, 1, 1, 128, 1, 1, 32, None),
        (7_000, 7_200, 13, PID, 3, 5, 16, 1, 1, 256, 1, 1, 20, 101)])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?, ?)", [
        (8_500, 8_700, 14, PID, 4_000, 2), (1_900, 2_000, 15, PID, 1_000, 1), (9_500, 9_600, 99, PID, 500, 1)])
    db.commit()
    db.close()


class RecordedFixtureTests(unittest.TestCase):
    def test_summary_reports_absent_tables_as_null(self):
        summary = q.event_summary(FIXTURE)
        self.assertEqual(summary["event_counts"], {"kernel": 1, "memcpy": None, "memset": None, "sync": 1,
                                                   "runtime_api": 3, "nvtx": 1})
        self.assertEqual(summary["absent_tables"], ["memcpy", "memset"])
        self.assertEqual(summary["time_span"]["duration_ns"], 558_312)

    def test_kernel_and_nvtx_stats(self):
        kernels = q.kernel_stats(FIXTURE)
        self.assertEqual((kernels["launches"], kernels["total_ns"]), (1, 1_024))
        only = kernels["kernels"][0]
        self.assertEqual(only["name"], "axpby_kernel")
        self.assertIn("axpby_kernel<", only["demangled_name"])
        self.assertEqual(only["block"], {"values": [[256, 1, 1]], "distinct": 1})
        self.assertEqual(q.nvtx_stats(FIXTURE)["ranges"][0]["text"], "agenticrl_generated")

    def test_missing_memcpy_table_is_not_a_measured_zero(self):
        self.assertEqual(q.memcpy_stats(FIXTURE), {"report": "memcpy", "directions": [], "reason": "table_absent"})

    def test_real_global_ids_attach_the_kernel_to_its_launch_call(self):
        result = q.query_tree(FIXTURE, "count_calls", event="axpby_kernel", root_name="cudaLaunchKernel_v7000")
        self.assertEqual((result["count"], result["total_ns"]), (1, 1_024))
        self.assertEqual(result["tree"]["gpu_events_without_host_parent"], 0)


class SyntheticExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = scratch()
        self.addCleanup(self.tmp.cleanup)
        self.sqlite = Path(self.tmp.name) / "trace.sqlite"
        synthetic_export(self.sqlite)

    def test_kernels_group_sort_and_limit(self):
        result = q.kernel_stats(self.sqlite)
        self.assertEqual((result["distinct_kernels"], result["launches"], result["total_ns"]), (2, 3, 1_600))
        self.assertEqual([k["name"] for k in result["kernels"]], ["expand_kernel", "shrink_kernel"])
        shrink = result["kernels"][1]
        self.assertEqual((shrink["count"], shrink["total_ns"], shrink["min_ns"], shrink["max_ns"]), (2, 600, 200, 400))
        self.assertEqual((shrink["mean_ns"], shrink["std_ns"], shrink["median_ns"]), (300.0, 100.0, 300.0))
        self.assertEqual(shrink["grid"], {"values": [[8, 1, 1], [16, 1, 1]], "distinct": 2})
        self.assertEqual(shrink["graph_node_launches"], 1)
        # A column the export lacks reads as None rather than a made-up value.
        self.assertEqual(shrink["static_shared_memory"], {"values": [None], "distinct": 1})
        by_count = q.kernel_stats(self.sqlite, top=1, sort="count")
        self.assertEqual([k["name"] for k in by_count["kernels"]], ["shrink_kernel"])
        with self.assertRaises(ValueError):
            q.kernel_stats(self.sqlite, sort="name")

    def test_nvtx_groups_by_text_and_domain_and_counts_marks(self):
        result = q.nvtx_stats(self.sqlite)
        self.assertEqual([(r["text"], r["count"], r["total_ns"]) for r in result["ranges"]],
                         [("outer", 1, 8_000), ("inner", 2, 5_000), ("other thread", 1, 1_000)])
        self.assertEqual(result["marks"], 1)
        only = q.nvtx_stats(self.sqlite, domain=3)
        self.assertEqual(([r["text"] for r in only["ranges"]], only["marks"]), (["other thread"], 0))

    def test_memcpy_groups_by_direction_with_bytes_and_bandwidth(self):
        by_direction = {d["direction"]: d for d in q.memcpy_stats(self.sqlite)["directions"]}
        self.assertEqual(sorted(by_direction), ["Device-to-Host", "Host-to-Device"])
        htod = by_direction["Host-to-Device"]
        self.assertEqual((htod["count"], htod["total_ns"], htod["total_bytes"]), (2, 200, 1_500))
        self.assertAlmostEqual(htod["bytes_per_second"], 1_500 / 200e-9)

    def test_tree_nests_host_intervals_and_hangs_gpu_work_under_its_call(self):
        root, info = q.build_tree(self.sqlite)
        self.assertEqual([n["name"] for n in root["children"]], ["outer", "other thread", "memcpy"])
        self.assertEqual(info["gpu_events_without_host_parent"], 1)   # the copy with correlation 99
        outer = root["children"][0]
        self.assertEqual([n["name"] for n in outer["children"]], ["inner", "inner", "cudaMemcpyAsync_v3020"])
        first_inner = outer["children"][0]
        self.assertEqual([n["name"] for n in first_inner["children"]], ["cudaLaunchKernel_v7000"] * 2)
        self.assertEqual([n["name"] for n in first_inner["children"][0]["children"]], ["shrink_kernel"])

    def test_tree_queries(self):
        top = q.query_tree(self.sqlite, "top_level")
        self.assertEqual([i["name"] for i in top["intervals"]], ["outer", "other thread", "memcpy"])
        longest = q.query_tree(self.sqlite, "most_time_consuming", root_name="outer")
        self.assertEqual((longest["event"]["name"], longest["event"]["duration_ns"]), ("inner", 3_000))
        kernel = q.query_tree(self.sqlite, "most_time_consuming", root_name="inner", types=("kernel", "nvtx"))
        self.assertEqual(kernel["event"]["name"], "expand_kernel")
        self.assertEqual(kernel["scopes_matched"], 2)
        counted = q.query_tree(self.sqlite, "count_calls", event="shrink_kernel", root_name="outer")
        self.assertEqual((counted["count"], counted["total_ns"]), (2, 600))
        shallow = q.query_tree(self.sqlite, "count_calls", event="shrink_kernel", root_name="outer", max_depth=2)
        self.assertEqual(shallow["count"], 0)   # kernels sit at depth 3: inner > launch call > kernel
        fuzzy = q.query_tree(self.sqlite, "count_calls", event="kernel", contains=True)
        self.assertEqual(fuzzy["count"], 3)     # the three kernels; matching is case-sensitive
        calls = q.query_tree(self.sqlite, "count_calls", event="LaunchKernel", contains=True)
        self.assertEqual(calls["count"], 3)     # the three launch calls

    def test_subtree_summary_and_thread_filter(self):
        summary = q.query_tree(self.sqlite, "subtree_summary", root_name="inner")
        self.assertEqual(summary["own"]["total_ns"], 5_000)
        self.assertEqual(summary["descendants_by_type"], {"kernel": {"count": 3, "total_ns": 1_600},
                                                          "trace": {"count": 3, "total_ns": 600}})
        self.assertEqual(summary["descendants_by_name"][0],
                         {"type": "kernel", "name": "expand_kernel", "count": 1, "total_ns": 1_000})
        worker = q.query_tree(self.sqlite, "top_level", thread=WORKER)
        self.assertEqual([i["name"] for i in worker["intervals"]], ["other thread"])
        self.assertEqual(worker["tree"]["nodes"], 3)   # range, copy call, copy

    def test_bad_requests_say_why(self):
        self.assertEqual(q.query_tree(self.sqlite, "top_level", root_name="absent")["reason"], "root_not_found")
        for kwargs in ({"query": "count_calls"}, {"query": "subtree_summary"}, {"query": "top_level", "types": ("gpu",)}):
            with self.assertRaises(ValueError):
                q.query_tree(self.sqlite, **kwargs)
        with self.assertRaises(FileNotFoundError):
            q.event_summary(self.sqlite.with_name("missing.sqlite"))

    def test_cli_prints_one_json_line_last(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            q.main(["kernels", str(self.sqlite), "--top", "1"])
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        result = json.loads(lines[-1])
        self.assertEqual((result["report"], len(result["kernels"]), result["sqlite"]), ("kernels", 1, str(self.sqlite)))


if __name__ == "__main__":
    unittest.main()
