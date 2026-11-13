"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["rotary_embedding_forward"]


def __getattr__(name):
    if name == "rotary_embedding_forward":
        from .rotary_embedding import rotary_embedding_forward
        return rotary_embedding_forward
    raise AttributeError(name)
