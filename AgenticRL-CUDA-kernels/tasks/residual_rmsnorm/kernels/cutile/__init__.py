"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["residual_rmsnorm_forward"]


def __getattr__(name):
    if name == "residual_rmsnorm_forward":
        from .residual_rmsnorm import residual_rmsnorm_forward
        return residual_rmsnorm_forward
    raise AttributeError(name)
