"""Lazy task-local cuTile export; importing the package does not load CUDA."""

__all__ = ["rotary_embedding_forward"]


def __getattr__(name):
    if name == "rotary_embedding_forward":
        from .rotary_embedding import rotary_embedding_forward
        return rotary_embedding_forward
    raise AttributeError(name)
