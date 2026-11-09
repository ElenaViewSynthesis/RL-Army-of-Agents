"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["cispo_clip_weight_forward"]


def __getattr__(name):
    if name == "cispo_clip_weight_forward":
        from .cispo_clip_weight import cispo_clip_weight_forward
        return cispo_clip_weight_forward
    raise AttributeError(name)
