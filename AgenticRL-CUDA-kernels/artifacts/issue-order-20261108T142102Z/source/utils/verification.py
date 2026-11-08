#!/usr/bin/env python3
"""Correctness verification for CUDA extension model."""

import argparse
import torch
import torch.nn.functional as F
from contextlib import contextmanager

from utils.backend import DEVICE


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, list):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, tuple):
        return tuple(transform_tensors(x, fn) for x in tensors)
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def check_equal(actual, expected):
    assert type(actual) == type(expected), f"{type(actual)=} != {type(expected)=}"
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), f"{len(actual)=} != {len(expected)=}"
        for x, y in zip(actual, expected):
            check_equal(x, y)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), "Output dictionary keys differ"
        for key, val in expected.items():
            assert key in actual, f"Missing key in output: {key}"
            check_equal(actual[key], val)
    elif isinstance(actual, (str, float, int)):
        assert actual == expected, f"{actual=} != {expected=}"
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    else:
        raise TypeError(f"Unsupported output type: {type(actual)}")


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith("_") and name not in excludes:
            originals[name] = attr

            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function torch.nn.functional.{__name} is not allowed in this context."
                )

            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


def initialize_models():
    from model import Model, get_init_inputs
    from model_new import ModelNew

    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    init_inputs = transform_tensors(init_inputs, lambda x: x.to(DEVICE))
    torch_model = Model(*init_inputs).eval().to(DEVICE)
    cuda_model = ModelNew(*init_inputs).eval().to(DEVICE)
    cuda_model.load_state_dict(torch_model.state_dict())
    return torch_model, cuda_model


def build_inputs():
    from model import get_inputs

    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]
    torch_inputs = transform_tensors(torch_inputs, lambda x: x.to(DEVICE))
    cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())
    return torch_inputs, cuda_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch_model, cuda_model = initialize_models()
    num_checks = 5

    with torch.no_grad():
        for i in range(num_checks):
            torch_inputs, cuda_inputs = build_inputs()
            torch_output = torch_model(*torch_inputs)
            with block_torch_functional():
                cuda_output = cuda_model(*cuda_inputs)
            check_equal(cuda_output, torch_output)
            print(f"[PASS] check {i + 1}/{num_checks}")

    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()
    print("[PASS] verify success")


if __name__ == "__main__":
    main()
