"""Kernel entry points; imported only inside kernel workers."""

__all__ = ["decode_attention_forward"]


def __getattr__(name):
    if name == "decode_attention_forward":
        from .decode_attention import decode_attention_forward
        return decode_attention_forward
    raise AttributeError(name)
