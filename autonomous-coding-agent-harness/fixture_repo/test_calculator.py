from app import safe_divide
from calculator import add, divide


def test_add() -> None:
    assert add(2, 3) == 5


def test_divide() -> None:
    assert divide(10, 2) == 5


def test_safe_divide() -> None:
    assert safe_divide(10, 2) == 5
