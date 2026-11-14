"""scripts/profile_tasks.py planning, checked without a GPU, Torch, nsys or Modal."""

import contextlib
import io
import json
import unittest

from scripts import profile_tasks as profile
from utils.task_catalog import get_task, select_cases


def dry_run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = profile.main([*argv, "--dry-run"])
    lines = out.getvalue().splitlines()
    return code, lines


class PlanTests(unittest.TestCase):
    def test_rows_cover_every_case_backend_and_tool_in_order(self):
        cases = [case.id for case in select_cases("relu_squared", "all")]
        rows = profile.plan(["relu_squared"], ["triton", "cuda"], "all", ["nsys", "ncu"])
        self.assertEqual(len(rows), len(cases) * 4)
        self.assertEqual([(r["case"], r["backend"], r["tool"]) for r in rows[:4]],
                         [(cases[0], "triton", "nsys"), (cases[0], "triton", "ncu"),
                          (cases[0], "cuda", "nsys"), (cases[0], "cuda", "ncu")])
        self.assertTrue(all(row["skip"] is None for row in rows))

    def test_case_sets_follow_the_catalog(self):
        for case_set in ("smoke", "acceptance"):
            with self.subTest(case_set=case_set):
                rows = profile.plan(["eagle_tree_step"], ["triton"], case_set, ["nsys"])
                self.assertEqual([r["case"] for r in rows], [c.id for c in select_cases("eagle_tree_step", case_set)])

    def test_backend_without_an_honest_fixture_is_skipped_with_a_reason(self):
        self.assertFalse(profile.fixture("relu_squared", "tilelang").is_file())
        rows = profile.plan(["relu_squared"], ["triton", "tilelang"], "smoke", ["nsys"])
        self.assertEqual([(r["backend"], r["skip"]) for r in rows],
                         [("triton", None), ("tilelang", "no_honest_fixture_for_backend")])

    def test_contract_only_task_is_skipped(self):
        self.assertTrue(get_task("fp8_gemm").contract_only)
        rows = profile.plan(["fp8_gemm"], ["triton"], "smoke", ["nsys"])
        self.assertTrue(rows and all(row["skip"] == "contract_only_task" for row in rows))

    def test_case_directory_names_are_plain_and_ordered(self):
        row = {"task": "rmsnorm", "case": "b32_h4096_float32", "backend": "triton", "tool": "nsys"}
        self.assertEqual(profile.case_name(7, row), "007-rmsnorm-b32_h4096_float32-triton-nsys")


class DryRunTests(unittest.TestCase):
    def test_dry_run_prints_one_json_plan_and_writes_nothing(self):
        code, lines = dry_run("--tasks", "relu_squared,rmsnorm", "--backends", "triton", "--tools", "nsys,ncu")
        self.assertEqual((code, len(lines)), (0, 1))
        plan = json.loads(lines[0])
        self.assertEqual((plan["tasks"], plan["tools"], plan["diagnostic_only"]),
                         (["relu_squared", "rmsnorm"], ["nsys", "ncu"], True))
        self.assertEqual((plan["to_run"], plan["skipped"]), (4, 0))

    def test_groups_expand_to_their_tasks(self):
        _, lines = dry_run("--tasks", "stage7")
        self.assertIn("relu_squared", json.loads(lines[0])["tasks"])

    def test_bad_requests_exit_before_any_work(self):
        for argv in (["--tasks", "no_such_task"], ["--tasks", "rmsnorm", "--backends", "fortran"],
                     ["--tasks", "rmsnorm", "--tools", "sanitizer"], ["--tasks", "rmsnorm", "--tools", "nsys,nsys"],
                     ["--tasks", "rmsnorm", "--nsys-profile", "everything"],
                     ["--tasks", "rmsnorm", "--timeout-s", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    profile.main([*argv, "--dry-run"])
                self.assertEqual(raised.exception.code, 2)

    def test_a_real_run_needs_an_output_directory(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            profile.main(["--tasks", "rmsnorm"])


if __name__ == "__main__":
    unittest.main()
