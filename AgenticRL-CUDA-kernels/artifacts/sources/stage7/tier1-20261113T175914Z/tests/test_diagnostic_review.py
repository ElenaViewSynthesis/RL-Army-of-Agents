import csv
import json
import shutil
import unittest
from pathlib import Path

from scripts.diagnose_gpu import plan
from tests.support import ROOT, scratch
from utils.backend import BACKENDS
from utils.diagnose import CHECKS, TOOLS
from utils.diagnostic_review import review_report, union_duration

FIXTURES = ROOT / "tests/fixtures/diagnostics"


class DiagnosticReviewTests(unittest.TestCase):
    def review(self, name, mutate=None):
        with scratch() as temporary:
            case = Path(temporary) / "case"
            shutil.copytree(FIXTURES / name, case)
            if mutate:
                mutate(case)
            review = review_report(case / "report.json")
            self.assertEqual(json.loads(json.dumps(review, allow_nan=False)), review)
            self.assertTrue(all(key.startswith("diag/") for key in review["result"]))
            return review

    def test_real_memcheck_and_racecheck_outputs_are_clean(self):
        for name in ("00-sanitizer-cuda-memcheck", "01-sanitizer-cuda-racecheck"):
            with self.subTest(name=name):
                review = self.review(name)
                self.assertEqual(review["result"]["diag/status"], "passed")
                self.assertEqual(review["result"]["diag/reported_errors"], 0)
                self.assertTrue(review["result"]["diag/evidence_reviewed"])

    def test_real_api_version_findings_never_become_a_clean_memory_check(self):
        review = self.review("16-sanitizer-cutedsl-memcheck")
        self.assertEqual(review["result"]["diag/status"], "failed")
        self.assertEqual(review["result"]["diag/reported_errors"], 18)
        self.assertEqual(review["result"]["diag/reason"], "cuda_api_version_findings")
        self.assertEqual(review["observations"]["cuda_api_version_mismatches"],
                         [{"requested": 13041, "driver": 13030}, {"requested": 13050, "driver": 13030}])

    def test_real_invalid_store_is_detected_without_accepting_the_control(self):
        review = self.review("20-sanitizer-triton-memcheck-invalid-access")
        self.assertEqual(review["result"]["diag/status"], "failed")
        self.assertEqual(review["result"]["diag/reported_errors"], 6)
        self.assertTrue(review["result"]["diag/negative_control_detected"])

    def test_synthetic_truncated_and_duplicate_summaries_fail_closed(self):
        def truncate(case):
            path = case / "sanitizer.log"
            path.write_text(path.read_text().splitlines()[0] + "\n")
        def duplicate(case):
            path = case / "sanitizer.log"
            path.write_text(path.read_text() + path.read_text().splitlines()[-1] + "\n")
        for mutate in (truncate, duplicate):
            with self.subTest(mutate=mutate.__name__):
                self.assertEqual(self.review("00-sanitizer-cuda-memcheck", mutate)["result"]["diag/status"],
                                 "failed")

    def test_synthetic_mismatched_gpu_identity_fails_closed(self):
        def mismatch(case):
            path = case / "report.json"
            report = json.loads(path.read_text())
            report["worker"]["gpu_uuid"] = "synthetic-other-gpu"
            path.write_text(json.dumps(report))
        self.assertEqual(self.review("00-sanitizer-cuda-memcheck", mismatch)["result"]["diag/status"],
                         "failed")

    def test_synthetic_zero_summary_does_not_hide_vendor_findings(self):
        def append_diagnostic(case):
            source = FIXTURES / "16-sanitizer-cutedsl-memcheck/sanitizer.log"
            line = next(line for line in source.read_text().splitlines() if "CUDA API Error:" in line)
            path = case / "sanitizer.log"
            path.write_text(path.read_text() + line + "\n")
        result = self.review("00-sanitizer-cuda-memcheck", append_diagnostic)["result"]
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("unavailable", "unrecognized_sanitizer_output"))

    def test_real_unavailable_tools_have_no_fabricated_kernel_counts(self):
        for name, reason in (("21-nsys-cuda-capture", "nsys_importer_unavailable"),
                             ("26-ncu-cuda-capture", "gpu_clock_control_unavailable")):
            with self.subTest(name=name):
                result = self.review(name)["result"]
                self.assertEqual((result["diag/status"], result["diag/reason"]), ("unavailable", reason))
                self.assertIsNone(result["diag/kernel_count"])

    def test_real_counter_csv_preserves_metric_strings_and_units(self):
        review = self.review("ncu-cuda-clock-none")
        self.assertEqual(review["result"]["diag/status"], "passed")
        self.assertEqual(review["result"]["diag/kernel_count"], 1)
        self.assertEqual(review["result"]["diag/ncu_metric_fields"], 562)
        self.assertEqual(review["observations"]["profiles"][0]["metrics"]["launch__registers_per_thread"],
                         {"value": "26", "unit": "register/thread"})
        self.assertNotIn("reward/value", review["result"])

    def test_synthetic_nonfinite_counter_value_fails_closed(self):
        def invalidate(case):
            path = case / "metrics.stdout.log"
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            rows[2][rows[0].index("gpu__time_duration.sum")] = "nan"
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
        result = self.review("ncu-cuda-clock-none", invalidate)["result"]
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("failed", "required_counter_metric_unavailable"))

    def test_real_recovered_timeline_has_one_kernel_and_unmeasured_transfers(self):
        review = self.review("nsys-cuda-recovered")
        self.assertEqual(review["result"]["diag/status"], "passed")
        self.assertEqual(review["result"]["diag/kernel_count"], 1)
        self.assertEqual(review["result"]["diag/nsys_kernel_busy_ns"], 1024)
        self.assertEqual(review["result"]["diag/nsys_kernel_gap_ns"], 0)
        self.assertIsNone(review["observations"]["transfer_duration_ns"])

    def test_synthetic_recovery_receipt_mismatch_fails_closed(self):
        def invalidate(case):
            path = case / "postprocess-v2.json"
            receipt = json.loads(path.read_text())
            receipt["base_receipt_sha256"] = "synthetic-mismatch"
            path.write_text(json.dumps(receipt))
        result = self.review("nsys-cuda-recovered", invalidate)["result"]
        self.assertEqual((result["diag/status"], result["diag/reason"]),
                         ("failed", "postprocess_provenance_mismatch"))

    def test_synthetic_overlapping_intervals_are_not_double_counted(self):
        self.assertEqual(union_duration([(2, 5), (4, 7), (9, 10)]), 6)
        self.assertEqual(union_duration([(2, 10), (4, 5)]), 8)

    def test_matrix_plan_is_sequential_and_includes_one_negative_control(self):
        jobs = plan(BACKENDS, TOOLS, CHECKS)
        self.assertEqual(len(jobs), 31)
        self.assertTrue(all(tool == "sanitizer" and not negative for tool, _, _, negative in jobs[:20]))
        self.assertEqual(jobs[20], ("sanitizer", "triton", "memcheck", True))
        self.assertTrue(all(tool == "nsys" for tool, _, _, _ in jobs[21:26]))
        self.assertTrue(all(tool == "ncu" for tool, _, _, _ in jobs[26:]))
