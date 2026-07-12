"""Deterministic seeding.

``seed_everything`` seeds Python's ``random`` plus numpy/torch when they are
installed (both are optional dependencies of downstream packages, not of core).
``derive_seed`` produces stable per-task sub-seeds so adding or reordering tasks
does not perturb the randomness of other tasks.
"""

import hashlib
import importlib.util
import random


def seed_everything(seed: int) -> int:
    """Seed all available RNG sources; returns the seed for logging."""
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy

        numpy.random.seed(seed % (2**32))
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
    return seed


def derive_seed(base_seed: int, *components: str) -> int:
    """Derive a stable sub-seed from a base seed and string components.

    Uses SHA-256 (not ``hash()``, which is salted per process) so results are
    reproducible across runs and machines.
    """
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode("utf-8"))
    for component in components:
        digest.update(b"\x00")
        digest.update(component.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")
