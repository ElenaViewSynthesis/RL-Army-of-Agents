# Stage 5 (EAGLE-4) figures and table

Publication assets built from the honest-solution timings of H200 run
`task-stage5-gpu-acceptance-20261110T002043Z` (commit `e48c177`; all 71 suite
rows passed; evidence in `artifacts/runs/task-stage5-gpu-acceptance-20261110T002043Z/`).

| File | What it is |
|---|---|
| `stage5-largest-honest.csv` | Source rows: one per task, backend and solution, with solution, eager and `torch.compile` profiler device time per call (µs) at each task's largest case. The same file backs the "Stage 5 kernel timings" dashboard. |
| `fig_stage5_speedup_vs_eager.svg` | Figure 1, the dashboard's dot plot: eager time ÷ solution time on a log scale. |
| `fig_stage5_time_per_call.svg` | Figure 2, the CSV rows as a paper figure: time per call for eager, `torch.compile`, CUDA and Triton on a log scale. |
| `table_stage5_largest.tex` | The dashboard's table as a LaTeX `booktabs` table (needs `booktabs` and `siunitx`). |
| `table_stage5_largest.md` | The same table in Markdown. |

Regenerate everything from the CSV (standard library only, no plotting
dependency):

```bash
python3 -m scripts.make_stage5_figures --csv assets/stage5/stage5-largest-honest.csv --out assets/stage5
```

For LaTeX, convert an SVG to PDF, for example with
`rsvg-convert -f pdf -o fig.pdf fig_stage5_time_per_call.svg`.

## Draft captions

**Figure 1.** Speed of the honest reference solutions relative to eager
PyTorch at each Stage 5 task's largest case on one NVIDIA H200 (log scale;
right of the line is faster). Squares are CUDA C++ and triangles Triton; the
hollow triangle is the alternate chain-replay solution for the GDN
recurrence. Tree convolutions, the GDN recurrence and greedy acceptance beat
eager. Candidate top-k and the two 32,768-token attention kernels are correct
but untuned, and slower than eager.

**Figure 2.** Time per call at each Stage 5 task's largest case (profiler
device time, log scale) for eager PyTorch (filled circle), `torch.compile`
(hollow circle) and the authored CUDA (square) and Triton (triangle)
solutions. The attention tasks use a 32,768-token context with 32 sequences,
64 query heads, 8 key/value heads, head dimension 128 and 64-node draft
trees.

Notes for the text: times are the reward profiler's per-call sum of CUDA
kernel durations after two empty priming captures. The reference solutions
demonstrate solvability and gate behaviour; they are not tuned kernels.
Kernel attribution is name-based.
