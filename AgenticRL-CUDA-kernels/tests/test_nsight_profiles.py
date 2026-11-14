"""Nsight profiles, NVTX ranges, per-kernel metrics and suite hooks: fake tools and SYNTHETIC evidence only.

No GPU, vendor tool or Modal job runs here. SQLite files and CSV rows built in
these tests are synthetic; the ncu CSV case starts from the real H200 fixture
tests/fixtures/diagnostics/ncu-cuda-clock-none (ncu 2026.2.1, AXPBY).
"""

import contextlib
import io
import json
import shutil
import sqlite3
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import ROOT, scratch
from utils import diagnose
from utils.diagnostic_review import range_summary, review_report

WORKER = ["python", "-c", "worker"]
OUT = Path("/out")
# The 2026-11-08 commands (run 20261108T200118Z), written out literally.
TRACE_ARGV = ["/bin/nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
              "--discard-environment=true", "--capture-range=nvtx",
              "--nvtx-capture=agenticrl_generated", "--capture-range-end=stop",
              "--export=sqlite", "--output", str(OUT / "capture"), *WORKER]
NCU_BASIC_ARGV = ["/bin/ncu", "--clock-control", "none", "--nvtx", "--nvtx-include", "agenticrl_generated/",
                  "--kernel-name-base", "demangled", "--kernel-name", "regex:k", "--launch-count", "1",
                  "--set", "basic", "--export", str(OUT / "capture"), *WORKER]


def nsys(profiles, device=None):
    return diagnose.tool_command("nsys", "/bin/nsys", WORKER, OUT, "memcheck", None,
                                 profiles=profiles, metrics_device=device)


class ProfileCommandTests(unittest.TestCase):
    def test_default_trace_and_ncu_basic_are_byte_identical_to_the_validated_commands(self):
        self.assertEqual(nsys(("trace",)), TRACE_ARGV)
        self.assertEqual(diagnose.tool_command("nsys", "/bin/nsys", WORKER, OUT, "memcheck", None), TRACE_ARGV)
        self.assertEqual(diagnose.tool_command("ncu", "/bin/ncu", WORKER, OUT, "memcheck", "k"), NCU_BASIC_ARGV)

    def test_each_profile_argv_is_pinned(self):
        def added(argv):
            return [arg for arg in argv if arg not in TRACE_ARGV]
        self.assertEqual(added(nsys(("stages",))),
                         ["--trace=cuda,nvtx,cublas,cudnn,osrt", "--nvtx-capture=agenticrl_rollout"])
        self.assertEqual(added(nsys(("pytorch",))), ["--pytorch=functions-trace,autograd-shapes-nvtx"])
        # Backtraces need CPU sampling, so this profile alone turns it on.
        self.assertEqual(added(nsys(("backtrace",))),
                         ["--sample=process-tree", "--cudabacktrace=all", "--python-backtrace=cuda"])
        self.assertEqual(added(nsys(("gpu-metrics",), device="3")), ["--gpu-metrics-devices=3"])
        self.assertEqual(added(nsys(("graphs",))), ["--cuda-graph-trace=node"])
        for argv in (nsys(("stages",)), nsys(("backtrace",))):
            self.assertEqual(argv[3], "--sample=process-tree" if "--cudabacktrace=all" in argv else "--sample=none")
            self.assertIn("--cpuctxsw=none", argv)
            self.assertEqual(argv[-len(WORKER) - 3:-len(WORKER)], ["--export=sqlite", "--output", str(OUT / "capture")])

    def test_profiles_compose_in_a_fixed_order_and_reject_unknown_names(self):
        composed = nsys(("gpu-metrics", "stages", "graphs"), device="0")
        self.assertEqual(composed, nsys(("stages", "graphs", "gpu-metrics"), device="0"))
        self.assertIn("--trace=cuda,nvtx,cublas,cudnn,osrt", composed)
        self.assertLess(composed.index("--gpu-metrics-devices=0"), composed.index("--cuda-graph-trace=node"))
        self.assertEqual(diagnose.parse_profiles("stages, gpu-metrics,stages"), ("stages", "gpu-metrics"))
        for bad in ("", "trace,bogus", ","):
            with self.assertRaises(ValueError):
                diagnose.parse_profiles(bad)
        with self.assertRaises(ValueError):
            nsys(("gpu-metrics",))  # never a guessed device index

    def test_ncu_presets(self):
        def preset(name):
            return diagnose.tool_command("ncu", "/bin/ncu", WORKER, OUT, "memcheck", "k", preset=name)
        roofline = preset("roofline")
        self.assertNotIn("--set", roofline)
        self.assertEqual([roofline[i + 1] for i, arg in enumerate(roofline) if arg == "--section"],
                         ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy"])
        self.assertEqual([preset("launch")[i + 1] for i, arg in enumerate(preset("launch")) if arg == "--section"],
                         ["LaunchStats"])
        for argv in (roofline, preset("launch")):
            self.assertIn("--launch-count", argv)
            self.assertEqual(argv[argv.index("--clock-control") + 1], "none")
        with self.assertRaises(ValueError):
            preset("full")

    def test_run_diagnostic_rejects_misplaced_options(self):
        with scratch() as tmp:
            tmp = Path(tmp)
            for kwargs in ({"tool": "ncu", "profile": "stages"}, {"tool": "nsys", "preset": "roofline"},
                           {"tool": "sanitizer", "preset": "launch"}, {"tool": "nsys", "profile": "bogus"}):
                with self.assertRaises(ValueError):
                    diagnose.run_diagnostic(kwargs.pop("tool"), tmp, tmp / "o", backend="triton", **kwargs)

    def test_task_case_variables_reach_the_worker_environment(self):
        env = {"CUDA_AGENT_TASK": "relu_squared", "CUDA_AGENT_TASK_CASE": "b64_h1024_float32",
               "WANDB_API_KEY": "secret", "PATH": "/usr/bin"}
        with scratch() as tmp, patch.dict("os.environ", env, clear=True):
            child = diagnose.diagnostic_environment(Path(tmp), "ncu")
        self.assertEqual((child["CUDA_AGENT_TASK"], child["CUDA_AGENT_TASK_CASE"]),
                         ("relu_squared", "b64_h1024_float32"))
        self.assertNotIn("WANDB_API_KEY", child)


class GpuMetricsDeviceTests(unittest.TestCase):
    def test_logical_ordinal_maps_through_cuda_visible_devices(self):
        from utils.gpu_lock import selected_device_token
        self.assertEqual(selected_device_token({"CUDA_AGENT_DEVICE": "cuda:1", "CUDA_VISIBLE_DEVICES": "2,3"}), "3")
        self.assertEqual(selected_device_token({"CUDA_AGENT_DEVICE": "cuda"}), "0")
        self.assertIsNone(selected_device_token({"CUDA_AGENT_DEVICE": "cpu"}))
        self.assertIsNone(selected_device_token({"CUDA_AGENT_DEVICE": "cuda:4", "CUDA_VISIBLE_DEVICES": "0"}))
        self.assertIsNone(selected_device_token({"CUDA_VISIBLE_DEVICES": "-1"}))
        self.assertEqual(diagnose.gpu_metrics_device({"CUDA_AGENT_DEVICE": "cuda:1", "CUDA_VISIBLE_DEVICES": "2,3"}), "3")

    def test_uuid_token_is_resolved_by_nvidia_smi_or_left_unresolved(self):
        env = {"CUDA_AGENT_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": "GPU-abc", "PATH": "/x"}
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            return types.SimpleNamespace(stdout="5\n")
        with patch.object(diagnose.shutil, "which", return_value="/x/nvidia-smi"):
            self.assertEqual(diagnose.gpu_metrics_device(env, run=run), "5")
            self.assertEqual(calls[0][-2:], ["-i", "GPU-abc"])

            def failing(argv, **kwargs):
                raise subprocess.CalledProcessError(1, argv)
            self.assertIsNone(diagnose.gpu_metrics_device(env, run=failing))
        with patch.object(diagnose.shutil, "which", return_value=None):
            self.assertIsNone(diagnose.gpu_metrics_device(env, run=run))


# ---------------------------------------------------------------------------
# SYNTHETIC Nsight Systems evidence.

GPU = {"available": True, "completed": True, "gpu_uuid": "GPU-synthetic", "gpu_name": "synthetic"}
CAPTURE_WORKER = {"diagnostic_only": True, "available": True, "completed": True, "mode": "capture",
                  "backend": "cuda", "gpu_uuid": "GPU-synthetic", "gpu_index": 0, "compute_capability": "9.0",
                  "captured_forwards": 1, "verification_checks": 2, "capture_range": "agenticrl_generated",
                  "warmup_iters": 5}
AUTHORED = "void axpby_kernel<256>(float *, const float *, const float *, float, int)"
TORCH = "void at::native::vectorized_elementwise_kernel<4, add>(int)"
CUBLAS = "sm90_xmma_gemm_bf16_cublas_kernel"


def write_nsys_case(case, profiles, *, ranges, kernels, metrics=None, agent_names=("axpby_kernel",)):
    case.mkdir(parents=True)
    argv = nsys(profiles, device="0" if "gpu-metrics" in profiles else None)
    (case / "capture.nsys-rep").write_bytes(b"synthetic report")
    (case / "tool.stdout.log").write_text("")
    (case / "tool.stderr.log").write_text("")
    db = sqlite3.connect(case / "capture.sqlite")
    db.executescript("""
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, nameId INTEGER, returnValue INTEGER);
    """)
    for name, (start, end) in ranges.items():
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL)", (start, end, name))
    for i, (start, end, name) in enumerate(kernels):
        db.execute("INSERT INTO StringIds VALUES (?,?)", (100 + i, name))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,0,?)", (start, end, 100 + i))
    if metrics is not None:
        db.executescript("""
            CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INTEGER, metricId INTEGER, metricName TEXT);
            CREATE TABLE GPU_METRICS(timestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL);
        """)
        db.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (1,?,?)",
                       [(1, "SM Active [Throughput %]"), (2, "DRAM Read Bandwidth [Throughput %]"),
                        (3, "DRAM Write Bandwidth [Throughput %]"), (4, "GR Active [Throughput %]")])
        db.executemany("INSERT INTO GPU_METRICS VALUES (?,1,?,?)", metrics)
    db.commit()
    db.close()
    report = {"schema": "gpu-diagnostics-v1", "diagnostic_only": True, "gpu_probe": dict(GPU, gpu_index=0),
              "worker": CAPTURE_WORKER, "agent_kernel_names": list(agent_names),
              "stages": [{"label": "tool", "argv": argv, "returncode": 0, "timed_out": False}],
              "result": {"diag/tool": "nsys", "diag/backend": "cuda", "diag/status": "captured",
                         "diag/reason": "raw_evidence_requires_review", "diag/gpu_uuid": "GPU-synthetic",
                         "diag/negative_control": False, "diag/check": None,
                         "diag/nsys_profiles": ",".join(profiles)}}
    (case / "report.json").write_text(json.dumps(report))
    return case / "report.json"


STAGE_RANGES = {"agenticrl_rollout": (0, 1000), "agenticrl_eager": (10, 100), "agenticrl_compile": (110, 400),
                "agenticrl_warmup": (410, 600), "agenticrl_generated": (700, 800)}
STAGE_KERNELS = [(20, 60, TORCH), (120, 220, CUBLAS), (230, 260, TORCH), (420, 440, AUTHORED),
                 (450, 470, AUTHORED), (710, 750, AUTHORED)]


class NsysReviewTests(unittest.TestCase):
    def test_stages_profile_summarises_every_range_by_provenance(self):
        with scratch() as tmp:
            review = review_report(write_nsys_case(Path(tmp) / "case", ("stages",),
                                                   ranges=STAGE_RANGES, kernels=STAGE_KERNELS))
        result = review["result"]
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("passed", "generated_forward_timeline_captured"))
        self.assertEqual(result["diag/kernel_count"], 1)  # the generated forward only, as before
        self.assertEqual(result["diag/nsys_kernel_busy_ns"], 40)
        self.assertEqual(result["diag/nsys/rollout/kernel_count"], 6)
        self.assertEqual(result["diag/nsys/eager/torch_ns"], 40)
        self.assertEqual(result["diag/nsys/compile/library_ns"], 100)  # cuBLAS counts as library on CUDA
        self.assertEqual(result["diag/nsys/compile/torch_ns"], 30)
        self.assertEqual(result["diag/nsys/warmup/agent_ns"], 40)
        self.assertEqual(result["diag/nsys/generated/agent_ns"], 40)
        self.assertEqual(result["diag/nsys/generated/library_ns"], 0)
        self.assertNotIn("diag/nsys/generated/sm_active_pct_mean", result)  # gpu-metrics did not run

    def test_default_trace_still_rejects_kernels_outside_the_generated_range(self):
        with scratch() as tmp:
            review = review_report(write_nsys_case(
                Path(tmp) / "case", ("trace",), ranges={"agenticrl_generated": (700, 800)},
                kernels=[(710, 750, AUTHORED), (900, 950, AUTHORED)]))
        self.assertEqual(review["result"]["diag/reason"], "invalid_or_out_of_range_kernel_events")

    def test_duplicate_ranges_fail_closed(self):
        with scratch() as tmp:
            case = Path(tmp) / "case"
            path = write_nsys_case(case, ("stages",), ranges=STAGE_RANGES, kernels=STAGE_KERNELS)
            db = sqlite3.connect(case / "capture.sqlite")
            db.execute("INSERT INTO NVTX_EVENTS VALUES (410, 600, 'agenticrl_warmup', NULL)")
            db.commit()
            db.close()
            self.assertEqual(review_report(path)["result"]["diag/reason"], "empty_or_ambiguous_timeline")

    def test_gpu_metrics_means_inside_the_generated_range(self):
        samples = [(710, 1, 40.0), (720, 1, 60.0), (710, 2, 10.0), (710, 3, 5.0), (720, 2, 20.0), (720, 3, 5.0),
                   (720, 4, 99.0), (900, 1, 100.0)]  # GR Active and out-of-range samples are ignored
        ranges = {"agenticrl_generated": (700, 800)}
        with scratch() as tmp:
            result = review_report(write_nsys_case(Path(tmp) / "a", ("gpu-metrics",), ranges=ranges,
                                                   kernels=[(710, 750, AUTHORED)], metrics=samples))["result"]
            self.assertEqual(result["diag/nsys/generated/sm_active_pct_mean"], 50.0)
            self.assertEqual(result["diag/nsys/generated/dram_throughput_pct_mean"], 20.0)
            self.assertIsNone(result["diag/nsys/gpu_metrics_reason"])
            absent = review_report(write_nsys_case(Path(tmp) / "b", ("gpu-metrics",), ranges=ranges,
                                                   kernels=[(710, 750, AUTHORED)]))["result"]
        self.assertEqual(absent["diag/status"], "passed")
        self.assertIsNone(absent["diag/nsys/generated/sm_active_pct_mean"])
        self.assertIsNone(absent["diag/nsys/generated/dram_throughput_pct_mean"])
        self.assertEqual(absent["diag/nsys/gpu_metrics_reason"], "gpu_metrics_tables_absent")

    def test_real_validated_timeline_still_passes(self):
        with scratch() as tmp:
            case = Path(tmp) / "case"
            shutil.copytree(ROOT / "tests/fixtures/diagnostics/nsys-cuda-recovered", case)
            result = review_report(case / "report.json")["result"]
        self.assertEqual(result["diag/status"], "passed")
        self.assertIsNone(result["diag/nsys/generated/agent_ns"])  # report predates recorded authored names


class NvtxProvenanceTests(unittest.TestCase):
    def test_nvtx_ranges_do_not_change_provenance_shares(self):
        from utils.provenance import summarize_events
        kernels = [(a, b, name) for a, b, name in STAGE_KERNELS]
        alone = range_summary({"agenticrl_rollout": (0, 1000)}, kernels, ["axpby_kernel"], "cuda")
        nested = range_summary(STAGE_RANGES, kernels, ["axpby_kernel"], "cuda")
        self.assertEqual(alone["agenticrl_rollout"], nested["agenticrl_rollout"])
        events = [(name, (b - a) / 1000) for a, b, name in kernels]
        expected = summarize_events(events, ["axpby_kernel"], "cuda")
        totals = nested["agenticrl_rollout"]
        total = sum(totals[f"{k}_ns"] for k in ("agent", "library", "torch", "memory"))
        for key in ("agent", "library", "torch", "memory"):
            self.assertAlmostEqual(totals[f"{key}_ns"] / total, expected["provenance/" + key])


# ---------------------------------------------------------------------------
# Nsight Compute per-kernel summary, starting from the real H200 CSV fixture.

class NcuReviewTests(unittest.TestCase):
    def review(self, mutate=None):
        with scratch() as tmp:
            case = Path(tmp) / "case"
            shutil.copytree(ROOT / "tests/fixtures/diagnostics/ncu-cuda-clock-none", case)
            report = json.loads((case / "report.json").read_text())
            report["agent_kernel_names"] = ["axpby_kernel"]
            if mutate:
                mutate(case, report)
            (case / "report.json").write_text(json.dumps(report))
            review = review_report(case / "report.json")
            markdown = (case / "ncu-summary.md").read_text() if (case / "ncu-summary.md").is_file() else None
            return review["result"], markdown

    def test_real_basic_export_yields_the_per_kernel_fields(self):
        result, markdown = self.review()
        self.assertEqual(result["diag/status"], "passed")
        self.assertEqual(result["diag/ncu_kernel"], "axpby_kernel")
        prefix = "diag/ncu/axpby_kernel/"
        self.assertEqual(result[prefix + "duration_us"], 2.208)
        self.assertEqual(result[prefix + "sm_throughput_pct"], 0.01986)
        self.assertEqual(result[prefix + "dram_throughput_pct"], 0.046897)
        self.assertAlmostEqual(result[prefix + "achieved_occupancy_pct"], 7.860803)
        self.assertEqual(result[prefix + "registers_per_thread"], 26)
        self.assertEqual((result[prefix + "shared_mem_static_bytes"], result[prefix + "shared_mem_dynamic_bytes"]), (0, 0))
        self.assertEqual((result[prefix + "grid_size"], result[prefix + "block_size"]), (1, 256))
        self.assertIn("| `duration_us` | 2.208 |", markdown)
        self.assertIn("Diagnostic only", markdown)

    def test_wrong_unit_becomes_null_and_missing_required_metric_fails(self):
        def rewrite(case, column, unit=None, drop=False):
            path = case / "metrics.stdout.log"
            import csv
            with path.open(newline="") as stream:
                header, units, values = list(csv.reader(stream))
            index = header.index(column)
            if drop:
                for row in (header, units, values):
                    del row[index]
            else:
                units[index] = unit
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerows([header, units, values])
            path.write_text(buffer.getvalue())

        result, _ = self.review(lambda case, report: rewrite(case, "launch__shared_mem_per_block_static", "Kbyte/block"))
        self.assertEqual(result["diag/status"], "passed")
        self.assertIsNone(result["diag/ncu/axpby_kernel/shared_mem_static_bytes"])
        result, _ = self.review(lambda case, report: rewrite(case, "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                                                             drop=True))
        self.assertEqual(result["diag/reason"], "required_counter_metric_unavailable")

    def test_presets_change_the_required_metrics(self):
        def preset(name):
            return lambda case, report: report["result"].__setitem__("diag/ncu_preset", name)
        self.assertEqual(self.review(preset("launch"))[0]["diag/status"], "passed")
        self.assertEqual(self.review(preset("roofline"))[0]["diag/status"], "passed")
        self.assertEqual(self.review(preset("bogus"))[0]["diag/reason"], "unknown_ncu_preset")

    def test_kernel_label_falls_back_to_the_base_name(self):
        from utils.diagnostic_review import kernel_label
        self.assertEqual(kernel_label(AUTHORED, []), "axpby_kernel")
        self.assertEqual(kernel_label("axpby_kernel_Kt3_f32", ["axpby_kernel", "axpby"]), "axpby_kernel")
        self.assertEqual(kernel_label("relu_squared_triton_chunks", ["relu_squared_triton_chunks"]),
                         "relu_squared_triton_chunks")


# ---------------------------------------------------------------------------
# NVTX ranges in the diagnostic worker, with torch faked.

class WorkerNvtxTests(unittest.TestCase):
    def run_worker(self, *extra):
        calls = []
        nvtx = types.SimpleNamespace(range_push=lambda name: calls.append(("push", name)),
                                     range_pop=lambda: calls.append(("pop",)))
        props = types.SimpleNamespace(name="synthetic", uuid="GPU-synthetic", major=9, minor=0)
        cuda = types.SimpleNamespace(is_available=lambda: True, set_device=lambda index: None,
                                     current_device=lambda: 0, get_device_properties=lambda device: props,
                                     synchronize=lambda: calls.append(("sync",)), nvtx=nvtx)
        fake_torch = types.SimpleNamespace(
            cuda=cuda, device=lambda name: types.SimpleNamespace(index=0), manual_seed=lambda seed: None,
            no_grad=contextlib.nullcontext, compile=lambda model: (calls.append(("compile",)) or model))
        verification = types.SimpleNamespace(
            block_torch_functional=contextlib.nullcontext, build_inputs=lambda: ((1,), (1,)),
            check_equal=lambda actual, expected: None,
            initialize_models=lambda: (lambda x: x, lambda x: x))
        from utils import diagnostic_worker
        with scratch() as tmp:
            evidence = Path(tmp) / "worker.json"
            argv = ["worker", "--mode", "capture", "--backend", "cuda", "--evidence", str(evidence), *extra]
            with patch.dict(sys.modules, {"torch": fake_torch, "utils.verification": verification}), \
                 patch.object(diagnostic_worker, "DEVICE", "cuda"), patch.object(sys, "argv", argv), \
                 contextlib.redirect_stdout(io.StringIO()):
                diagnostic_worker.main()
            return calls, json.loads(evidence.read_text())

    def test_ranges_and_nesting(self):
        calls, record = self.run_worker()
        nvtx = [call for call in calls if call[0] in ("push", "pop")]
        self.assertEqual(nvtx, [("push", "agenticrl_rollout"), ("push", "agenticrl_eager"), ("pop",),
                                ("push", "agenticrl_warmup"), ("pop",), ("push", "agenticrl_generated"), ("pop",),
                                ("pop",)])
        self.assertEqual(record["nvtx_ranges"], ["agenticrl_rollout", "agenticrl_eager", "agenticrl_warmup",
                                                 "agenticrl_generated"])
        self.assertEqual((record["captured_forwards"], record["capture_range"], record["compile_phase"]),
                         (1, "agenticrl_generated", False))
        self.assertNotIn(("compile",), calls)
        generated = calls.index(("push", "agenticrl_generated"))
        self.assertEqual(calls[generated + 1:generated + 3], [("sync",), ("pop",)])  # exactly one forward inside

    def test_compile_phase_only_when_requested(self):
        calls, record = self.run_worker("--with-compile")
        self.assertIn(("push", "agenticrl_compile"), calls)
        self.assertIn(("compile",), calls)
        self.assertTrue(record["compile_phase"])
        self.assertLess(calls.index(("push", "agenticrl_compile")), calls.index(("push", "agenticrl_warmup")))


# ---------------------------------------------------------------------------
# Suite hooks: run_diagnostic and the GPU lock are mocked.

class FakeLock:
    busy = False

    def __init__(self, env, enabled, timeout_s):
        self.env = env

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BusyLock(FakeLock):
    busy = True


class SuiteHookTests(unittest.TestCase):
    ROW = {"suite/task": "relu_squared", "suite/case": "b64_h1024_float32", "suite/stage": 7,
           "suite/backend": "triton", "suite/fixture": "honest", "suite/status": "passed",
           "suite/reason": "expected_outcome", "reward/value": 2.0, "reward/base": 2,
           "reward/reason": "faster_than_eager", "reward/overlong_penalty": 0.0, "env/t_gen_us": 1.0}

    def fake_run(self, calls):
        def run(tool, workdir, out, **kwargs):
            import os
            calls.append((tool, kwargs, os.environ.get("CUDA_AGENT_TASK"), os.environ.get("CUDA_AGENT_TASK_CASE")))
            if tool == "ncu":
                return {"diag/status": "passed", "diag/reason": "selected_kernel_counters_captured",
                        "diag/review_json": str(out / "reviewed.json"), "diag/ncu_kernel": "relu_squared_triton_chunks",
                        "diag/ncu/relu_squared_triton_chunks/duration_us": 0.9,
                        "diag/ncu/relu_squared_triton_chunks/grid_size": 64}
            return {"diag/status": "passed", "diag/reason": "sanitizer_clean", "diag/reported_errors": 0,
                    "diag/review_json": str(out / "reviewed.json")}
        return run

    def attach(self, row, workdir, lock=FakeLock, **kwargs):
        from utils.task_suite import attach_diagnostics
        calls = []
        result = attach_diagnostics(dict(row), workdir, run=self.fake_run(calls), lock=lock, **kwargs)
        return result, calls

    def test_honest_row_gets_ncu_and_sanitizer_fields_and_an_unchanged_reward(self):
        with scratch() as tmp:
            workdir = Path(tmp)
            (workdir / "model_new.py").write_text("# synthetic\n")
            result, calls = self.attach(self.ROW, workdir, ncu="roofline", sanitizer=("memcheck", "racecheck"))
        self.assertEqual([call[0] for call in calls], ["ncu", "sanitizer", "sanitizer"])
        self.assertEqual(calls[0][1], {"backend": "triton", "timeout_s": 180, "preset": "roofline"})
        self.assertEqual(calls[2][1]["check"], "racecheck")
        self.assertTrue(all(call[2:] == ("relu_squared", "b64_h1024_float32") for call in calls))
        self.assertEqual(result["ncu/status"], "passed")
        self.assertEqual(result["ncu/kernel"], "relu_squared_triton_chunks")
        self.assertEqual(result["ncu/relu_squared_triton_chunks/duration_us"], 0.9)
        self.assertEqual((result["sanitizer/memcheck/status"], result["sanitizer/memcheck/errors"]), ("passed", 0))
        self.assertIn("sanitizer/racecheck/report_json", result)
        for key, value in self.ROW.items():
            self.assertEqual(result[key], value)  # diagnostics never change reward, env or suite fields
        added = set(result) - set(self.ROW)
        self.assertTrue(all(key.startswith(("ncu/", "sanitizer/")) for key in added))
        json.dumps(result, allow_nan=False)

    def test_skips_carry_a_reason_and_never_run_the_tool(self):
        with scratch() as tmp:
            workdir = Path(tmp)
            (workdir / "model_new.py").write_text("# synthetic\n")
            cases = (({**self.ROW, "suite/fixture": "hack"}, workdir, {}, "not_honest_fixture"),
                     (self.ROW, workdir, {"device": "cpu"}, "cpu_device"),
                     (self.ROW, workdir / "missing", {}, "row_not_evaluated"),
                     (self.ROW, workdir, {"lock": BusyLock}, "gpu_busy"))
            for row, path, kwargs, reason in cases:
                with self.subTest(reason=reason):
                    result, calls = self.attach(row, path, ncu="basic", sanitizer=("memcheck",), **kwargs)
                    self.assertEqual(calls, [])
                    self.assertEqual((result["ncu/status"], result["ncu/reason"]), ("skipped", reason))
                    self.assertEqual(result["sanitizer/memcheck/reason"], reason)
                    self.assertEqual(result["reward/value"], self.ROW["reward/value"])

    def test_no_hooks_add_no_keys(self):
        with scratch() as tmp:
            result, calls = self.attach(self.ROW, Path(tmp))
        self.assertEqual((result, calls), (self.ROW, []))

    def test_new_row_keys_have_one_adr_classification(self):
        from tracking.contract import classify
        for key in ("ncu/status", "ncu/reason", "ncu/preset", "ncu/report_json", "ncu/kernel",
                    "ncu/relu_squared_triton_chunks/duration_us", "sanitizer/memcheck/status",
                    "sanitizer/memcheck/reason", "sanitizer/memcheck/errors", "sanitizer/memcheck/report_json"):
            self.assertEqual(classify("result", key), "table")

    def test_run_suite_and_cli_reject_bad_selections(self):
        from utils.task_suite import run_suite
        with scratch() as tmp:
            for kwargs in ({"ncu": "full"}, {"sanitizer": ("leakcheck",)}):
                with self.assertRaises(ValueError):
                    run_suite(["relu_squared"], Path(tmp) / "out", device="cpu", **kwargs)
            self.assertFalse((Path(tmp) / "out").exists())
        proc = subprocess.run([sys.executable, "-m", "scripts.validate", "--ncu", "basic"], cwd=ROOT,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--suite", proc.stderr)

    def test_modal_runner_forwards_the_hooks(self):
        import ast
        tree = ast.parse((ROOT / "cloud/modal_tasks.py").read_text())
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for name in ("evaluate", "main"):
            self.assertTrue({"ncu", "sanitizer"} <= {arg.arg for arg in functions[name].args.args})
        source = (ROOT / "cloud/modal_tasks.py").read_text()
        self.assertIn('command += ["--ncu", ncu]', source)
        self.assertIn('command += ["--sanitizer", sanitizer]', source)
        main = functions["main"]
        main.decorator_list = []
        calls = []
        namespace = {"evaluate": types.SimpleNamespace(
                         spawn=lambda *a, **k: calls.append((a, k)) or types.SimpleNamespace(object_id="synthetic")),
                     "json": json, "wait_for_call": lambda call: {"synthetic": True}}
        exec(compile(ast.Module(body=[main], type_ignores=[]), "modal_tasks_main", "exec"), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["main"](run_id="synthetic-no-launch")
            namespace["main"](run_id="synthetic-no-launch", ncu="roofline", sanitizer="memcheck")
        self.assertEqual(calls[0][1], {})  # default call unchanged
        self.assertEqual(calls[1][1], {"ncu": "roofline", "sanitizer": "memcheck"})


if __name__ == "__main__":
    unittest.main()
