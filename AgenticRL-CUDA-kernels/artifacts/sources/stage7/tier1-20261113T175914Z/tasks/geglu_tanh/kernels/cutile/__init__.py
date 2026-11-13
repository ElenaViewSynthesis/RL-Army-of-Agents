"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["geglu_tanh_forward"]


def __getattr__(name):
    if name == "geglu_tanh_forward":
        from .geglu_tanh import geglu_tanh_forward
        return geglu_tanh_forward
    raise AttributeError(name)
