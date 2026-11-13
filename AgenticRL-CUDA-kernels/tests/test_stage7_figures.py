"""The Stage 7 README figure renders from the committed acceptance evidence (no GPU, no Torch)."""

import unittest

from scripts import make_stage7_figures as figures


class Stage7FigureTests(unittest.TestCase):
    def test_largest_case_chart_has_every_honest_kernel(self):
        svg, rows = figures.largest_case_svg()
        self.assertEqual(rows, 12)
        self.assertIn(figures.RUN, svg)
        self.assertEqual(svg.count('<circle '), 12 + 1)  # one generated time per task and backend, plus the legend
        self.assertEqual(svg.count(">faster than both</text>"), 1)  # Triton RoPE only
        self.assertEqual(svg.count(">faster than eager</text>"), 11)
        self.assertIn("rotary_embedding · Triton: generated 346.9 µs", svg)

    def test_missing_honest_row_is_an_error(self):
        rows = [r for r in figures.load_rows() if not (r["suite/fixture"] == "honest"
                                                       and r["suite/backend"] == "cutile")]
        with self.assertRaises(KeyError):
            figures.honest_rows(rows)


if __name__ == "__main__":
    unittest.main()
