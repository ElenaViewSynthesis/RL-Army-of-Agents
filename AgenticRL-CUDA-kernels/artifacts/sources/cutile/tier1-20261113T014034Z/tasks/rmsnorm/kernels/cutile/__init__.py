"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["rmsnorm_forward"]


def __getattr__(name):
    if name == "rmsnorm_forward":
        from .rmsnorm import rmsnorm_forward
        return rmsnorm_forward
    raise AttributeError(name)
