"""New-task integer equality; stock floating-point comparison is unchanged."""

from utils import verification


def _exact_integers(actual, expected):
    if isinstance(expected, verification.torch.Tensor):
        if not expected.is_floating_point() and not expected.is_complex():
            verification.torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, (list, tuple)):
        for value, target in zip(actual, expected):
            _exact_integers(value, target)
    elif isinstance(expected, dict):
        for key in expected:
            _exact_integers(actual[key], expected[key])


def check_equal(actual, expected):
    verification.check_equal(actual, expected)
    _exact_integers(actual, expected)


if __name__ == "__main__":
    verification.main(compare=check_equal)
