"""Conservative, Torch-free attribution of profiler device events.

Shares use total device time; the reward fraction excludes memory time.
Unknown names count against provenance. Source names are attribution hints,
not a security boundary against a policy deliberately spoofing kernel names.
"""

import math
import re
from collections import defaultdict

from utils.backend import BACKENDS, creditable_kernel_name


# Change this identifier whenever attribution semantics change.
PROVENANCE_RULESET = "anchored-namespaces-generic-v1"


def classify_kernel(name, agent_names, backend):
    # Namespace rejection precedes authored matching; "cutlass" must follow it
    # so CuTe DSL's authored device symbols retain credit.
    lower = name.lower()
    if "memcpy" in lower or "memset" in lower:
        return "memory"
    if (any(marker in lower for marker in ("at::native", "aten::", "c10::"))
            or re.search(r"^triton_(poi|per|red|tem)_fused", lower)):
        return "torch"
    spec = BACKENDS.get(backend)
    if spec is not None and any(
            creditable_kernel_name(agent_name)
            and re.search(spec.symbol_pattern.format(name=re.escape(agent_name)), name)
            for agent_name in agent_names):
        return "agent"
    if any(lib in lower for lib in ("cublas", "cutlass", "cudnn")):
        spec = BACKENDS.get(backend)
        return "library" if spec is not None and spec.library_allowed else "torch"
    return "torch"


def summarize_events(events, agent_names, backend, iters=1):
    """events are (name, device_time_us) pairs for leaf CUDA events only."""
    if backend not in BACKENDS or iters <= 0:
        raise ValueError("Invalid backend or iteration count")
    totals = {k: 0.0 for k in ("agent", "library", "torch", "memory")}
    kernels = defaultdict(float)
    for name, duration in events:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Profiler duration must be finite and non-negative")
        kernels[name] += duration / iters
        totals[classify_kernel(name, agent_names, backend)] += duration / iters
    total = sum(totals.values())
    non_memory = sum(totals[k] for k in ("agent", "library", "torch"))
    result = {"t_gen_us": total,
              "provenance": (totals["agent"] + totals["library"]) / non_memory if non_memory > 0 else 0.0}
    result.update({"provenance/" + k: v / total if total else 0.0 for k, v in totals.items()})
    result.update({"kernel_us/" + name: kernels[name] for name in sorted(kernels)})
    return result
