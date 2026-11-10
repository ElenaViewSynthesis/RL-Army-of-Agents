# ADR 0001 — CISPO as a third algorithm preset

- **Status:** accepted (2026-11-07)
- **Scope:** `utils/config.py`, `utils/grpo.py` consumers, `scripts/validate.py`, tests, sweep config, docs
- **GPU evidence:** Modal H200 run `20261107T140018Z` (`Ran 31 tests, OK`), see VALIDATION.md "CISPO GPU batch-ranking validation"

## Context

The environment already exposed GRPO and DAPO as `RolloutConfig` presets. The
environment owns the critic-free pieces (group-relative advantages, tied-group
filtering, overlong-turn shaping); the clip range, loss aggregation and KL are
recorded as configuration for the trainer, which this repo does not contain.
CISPO (MiniMax-M1) reuses GRPO's advantages but changes the loss to a
policy-gradient term weighted by a clipped, stop-gradient importance-sampling
(IS) weight, so it fits the same split with one new preset.

## Decisions

1. **One algorithm list.** `ALGOS = ("grpo", "dapo", "cispo")` in
   `utils/config.py` is the single source for both `RolloutConfig` validation
   and the `--algo` CLI choices. `scripts/validate.py` and the three test loops
   iterate `for algo in ALGOS:` instead of a literal pair, so a future preset
   is covered automatically.
2. **`preset("cispo")`** = `clip_low=1.0`, `clip_high=4.0`, `loss_agg="token"`,
   `kl_coef=0.0`, `filter_tied_groups=True`, `overlong_buffer_turns=4`.
   In this repo's `[1 − clip_low, 1 + clip_high]` convention that keeps the IS
   weight within `[0, 5]`. Following the paper, the preset adopts DAPO's
   dynamic sampling (tied-group filtering) and overlong penalty, with no KL.
3. **`clip_low = 1.0` is legal for CISPO only.** It means "no lower bound", as
   the paper does (ε_low set large). GRPO and DAPO keep `clip_low < 1`.
4. **Upper bound is not from the paper.** MiniMax-M1 publishes no ε_high; it
   states it tuned only ε_high. The value follows ms-swift's CISPO example
   (`--epsilon_high 5.0`, i.e. `min(r, 5.0)`), which cites ScaleRL; ScaleRL
   fixes the lower clip at 0 and reports CISPO "largely insensitive" to the
   upper clip (§A.17.2). Override with `--clip-high` when a trainer's observed
   IS ratios argue for another cap.
5. **The CISPO loss is not computed here.** The environment produces
   advantages, filtering and shaping; the trainer applies
   `sg(clip(r, 1 − clip_low, 1 + clip_high)) · Â · log π` averaged over all
   tokens. Clip/KL settings stay trainer-owned exactly as for GRPO and DAPO.
   A Torch-free reference function for the per-token weight (pure `math`,
   outside `reward.py`/`rollout_env.py`) is an accepted follow-up.
6. **Sweep.** `sweeps/rollout_sweep.yaml` lists `cispo` under `algo`, raising
   the grid from 32 to 48 runs (3 × 4 × 2 × 2); set `wandb agent --count`
   accordingly.

## Consequences

- Result keys, the reward rules and `t_*_us` are unchanged; only the preset
  table and validation of `clip_low` grew.
- Tests: `test_cispo_preset_and_is_weight_bounds` checks the preset values,
  `--algo cispo` on the CLI and the `clip_low` rules; the two `..._both_presets`
  tests became `..._every_preset`; the rollout-env overlong test also checks
  `("cispo", .75)`. Suite: `Ran 115 tests, OK (skipped=32)` on WSL. On native
  Windows 5 `rollout_env` process tests error because `os.killpg` does not
  exist there — pre-existing and unrelated (Linux/WSL2 is required).
- Docs: README config table gained a CISPO column, the compact explanation and
  the three sources under "Background reading"; the four `SKILL_*.md` files
  and the README say "DAPO and CISPO" for tied-group filtering and the overlong
  penalty; KNOWN_ISSUES notes the KL difference for controlled comparisons.
- GPU: `test_every_preset_ranks_the_hack_last` passed for the CUDA/Triton group
  and all three DSL groups on H200 (`build_batch` over measured rewards; not a
  trainer test). Measured CISPO group advantages: honest 0.7071 / 0.7071, hack
  −1.4142, `adv/frac_dropped` 0.

## Sources

- MiniMax-M1 — <https://arxiv.org/html/2506.13585v1>
- ms-swift CISPO docs — <https://swift.readthedocs.io/en/v4.4/Instruction/GRPO/AdvancedResearch/CISPO.html>
- ScaleRL — <https://arxiv.org/html/2510.13786>
