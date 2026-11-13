"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["relu_squared_forward"]


def __getattr__(name):
    if name == "relu_squared_forward":
        from .relu_squared import relu_squared_forward
        return relu_squared_forward
    raise AttributeError(name)
