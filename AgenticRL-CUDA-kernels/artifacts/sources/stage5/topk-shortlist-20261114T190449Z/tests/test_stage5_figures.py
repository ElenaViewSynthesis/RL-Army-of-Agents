"""The Stage 5 publication figures build from the committed CSV with the stdlib only."""

import subprocess
import sys
import unittest
import xml.dom.minidom
from pathlib import Path

from tests.support import ROOT, scratch
from scripts import make_stage5_figures as figures

CSV = ROOT / "assets" / "stage5" / "stage5-largest-honest.csv"


class Stage5FigureTests(unittest.TestCase):
    def test_outputs_are_valid_and_complete(self):
        rows = figures.load(CSV)
        self.assertEqual(len(rows), 19)
        self.assertEqual(len(figures.tasks_in_order(rows)), 9)
        for svg in (figures.speedup_figure(rows), figures.time_figure(rows)):
            document = xml.dom.minidom.parseString(svg)
            self.assertEqual(document.documentElement.getAttribute("role"), "img")
            self.assertNotIn("nan", svg.lower())
        table = figures.markdown_table(rows)
        self.assertEqual(table.count("\n"), 2 + len(rows))
        self.assertIn("141.5× slower", table)
        self.assertIn(r"\begin{tabular}", figures.latex_table(rows))

    def test_cli_writes_all_files_without_plotting_libraries(self):
        with scratch() as tmp:
            code = ("import sys; from scripts import make_stage5_figures as f; "
                    f"f.main(['--csv', {str(CSV)!r}, '--out', {tmp!r}]); "
                    "assert 'matplotlib' not in sys.modules and 'torch' not in sys.modules")
            proc = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["fig_stage5_speedup_vs_eager.svg", "fig_stage5_time_per_call.svg",
                              "table_stage5_largest.md", "table_stage5_largest.tex"])


if __name__ == "__main__":
    unittest.main()
