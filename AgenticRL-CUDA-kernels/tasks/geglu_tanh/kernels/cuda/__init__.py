"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["geglu_tanh_forward"]


def __getattr__(name):
    if name == "geglu_tanh_forward":
        from .geglu_tanh import geglu_tanh_forward
        return geglu_tanh_forward
    raise AttributeError(name)
