"""cuTile kernel entry points; imported only inside kernel workers."""

__all__ = ["axpby_forward"]


def __getattr__(name):
    if name == "axpby_forward":
        from .axpby import axpby_forward
        return axpby_forward
    raise AttributeError(name)
