"""Intentional Torch-op negative controls; never imported by honest kernels."""


def wrong_output(output):
    import torch

    if isinstance(output, torch.Tensor):
        return torch.full_like(output, float("nan") if output.is_floating_point() else -2147483648)
    if isinstance(output, tuple):
        return tuple(wrong_output(value) for value in output)
    if isinstance(output, list):
        return [wrong_output(value) for value in output]
    if isinstance(output, dict):
        return {key: wrong_output(value) for key, value in output.items()}
    raise TypeError(f"Unsupported negative-control output {type(output)}")
