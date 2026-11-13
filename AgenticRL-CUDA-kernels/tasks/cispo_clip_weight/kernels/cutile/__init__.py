"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["cispo_clip_weight_forward"]


def __getattr__(name):
    if name == "cispo_clip_weight_forward":
        from .cispo_clip_weight import cispo_clip_weight_forward
        return cispo_clip_weight_forward
    raise AttributeError(name)
