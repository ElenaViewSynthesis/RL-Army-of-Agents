"""Stage 6 README figures render from the committed run evidence (no GPU, no Torch)."""

import unittest

from scripts import make_stage6_figures as figures


class Stage6FigureTests(unittest.TestCase):
    def test_coverage_matrix_matches_the_stratified_summary(self):
        svg, rows = figures.coverage_svg()
        self.assertEqual(rows, 336)
        self.assertIn("Coverage matrix · Stratified", svg)
        self.assertIn("312 passed, 0 failed, 24 unavailable (24 planned not-applicable)", svg)
        self.assertEqual(svg.count(">n/a</text>"), 24)
        self.assertNotIn(">✗</text>", svg)  # no failed rows in this run

    def test_slot_ranks_cover_every_case_and_mark_padding(self):
        svg, cases = figures.slot_ranks_svg()
        self.assertEqual(cases, 14)
        self.assertIn('fill="url(#pad)"', svg)
        self.assertIn("slot 0: rank 8, padded to 16", svg)

    def test_cell_encoding(self):
        honest = {"suite/status": "passed", "suite/fixture": "honest", "reward/reason": "faster_than_eager"}
        caught = {"suite/status": "passed", "suite/fixture": "wrong"}
        na = {"suite/status": "unavailable", "suite/fixture": "wrong_skip",
              "suite/reason": "control_not_applicable_no_skipped_tokens"}
        other = {"suite/status": "unavailable", "suite/fixture": "honest", "suite/reason": "cuda_unavailable"}
        self.assertEqual(figures.cell(honest)[:2], (figures.SEQ[1], "2"))
        self.assertEqual(figures.cell(caught)[:2], (figures.GOOD, "✓"))
        self.assertEqual(figures.cell(na)[1], "n/a")
        self.assertEqual(figures.cell(other)[:2], (figures.WARN, "?"))


if __name__ == "__main__":
    unittest.main()
