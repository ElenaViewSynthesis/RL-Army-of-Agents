"""Pure-Python milestone reward, provenance gate, and optional turn shaping.

reward_base is always in {-1, 1, 2, 3}, including gate failures. reward is
returned as a float; only DAPO-style shaping makes it non-integral. Timing
uses the upstream criterion t_gen < (1 - margin) * baseline (strict >5%).
"""

import math
from collections.abc import Mapping


def compute_reward(verify_ok, t_eager, t_compile, t_gen, provenance, cfg, turns_used=None):
    if turns_used is not None and (not isinstance(turns_used, int) or turns_used < 0):
        raise ValueError("turns_used must be a non-negative integer or None")

    def result(base, reason, penalty=0.0):
        return {"reward": float(base - penalty), "reward_base": base,
                "reason": reason, "overlong_penalty": float(penalty)}

    if turns_used is not None and turns_used > cfg.max_turns:
        return result(-1, "max_turns")
    if not verify_ok:
        return result(-1, "correctness")
    if isinstance(provenance, Mapping):
        provenance = provenance.get("agent", 0.0) + provenance.get("library", 0.0)
    if provenance is None:
        return result(-1, "unmeasured")
    if (not math.isfinite(provenance) or not 0 <= provenance <= 1
            or provenance < cfg.provenance_threshold):
        return result(-1, "provenance")
    if not all(t is not None and math.isfinite(t) and t > 0 for t in (t_eager, t_compile, t_gen)):
        return result(-1, "timing")

    faster_eager = t_gen < (1 - cfg.speedup_margin) * t_eager
    faster_compile = t_gen < (1 - cfg.speedup_margin) * t_compile
    base = 3 if faster_eager and faster_compile else (2 if faster_eager else 1)
    reason = {1: "correct", 2: "faster_than_eager", 3: "faster_than_both"}[base]
    penalty = 0.0
    if turns_used is not None and cfg.overlong_buffer_turns:
        start = cfg.max_turns - cfg.overlong_buffer_turns
        penalty = max(0.0, (turns_used - start) / cfg.overlong_buffer_turns)
    return result(base, reason, penalty)
