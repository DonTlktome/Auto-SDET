"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorial import factorial, factorial_recursive  # noqa: E402


# --------------------------------------------------------------------------- #
# factorial() — happy path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "number, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
        (6, 720),
        (10, 3628800),
        (20, 2432902008176640000),
    ],
)
def test_factorial_known_values(number, expected):
    """factorial() returns the mathematically correct value for small inputs."""
    assert factorial(number) == expected


def test_factorial_matches_math_factorial():
    """factorial() agrees with math.factorial for a range of inputs."""
    assert all(factorial(i) == math.factorial(i) for i in range(30))


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1 (loop body executes exactly once)."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


# --------------------------------------------------------------------------- #
# factorial() — edge cases / errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [0.1, 1.5, 2.0001, -0.5, 3.7])
def test_factorial_rejects_non_integral_floats(number):
    """factorial() raises ValueError for non-integral float inputs."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(number)


@pytest.mark.parametrize("number", [-1, -2, -100])
def test_factorial_rejects_negative_values(number):
    """factorial() raises ValueError for negative inputs."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(number)


def test_factorial_integral_float_raises_type_error():
    """An integral float passes the value check but fails on `range()`.

    The implementation compares against ``int(number)`` and then iterates with
    ``range(1, number + 1)``, which requires an ``int``.  An integral float
    such as ``5.0`` therefore reaches ``range()`` and raises ``TypeError``.
    """
    with pytest.raises(TypeError):
        factorial(5.0)


def test_factorial_rejects_non_numeric_string():
    """factorial() raises TypeError when given a non-numeric string."""
    with pytest.raises((TypeError, ValueError)):
        factorial("5")


# --------------------------------------------------------------------------- #
# factorial_recursive() — happy path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
        (6, 720),
        (10, 3628800),
    ],
)
def test_factorial_recursive_known_values(n, expected):
    """factorial_recursive() returns the mathematically correct value."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial():
    """factorial_recursive() agrees with math.factorial for a range of inputs."""
    assert all(factorial_recursive(i) == math.factorial(i) for i in range(20))


def test_factorial_recursive_base_cases():
    """Boundary: both base cases (0 and 1) return 1."""
    assert factorial_recursive(0) == 1
    assert factorial_recursive(1) == 1


def test_factorial_recursive_agrees_with_iterative():
    """The recursive and iterative implementations produce identical results."""
    assert all(factorial_recursive(i) == factorial(i) for i in range(25))


# --------------------------------------------------------------------------- #
# factorial_recursive() — edge cases / errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0.1, 1.5, 2.0, -0.5])
def test_factorial_recursive_rejects_non_int_types(n):
    """factorial_recursive() raises ValueError for any non-int type, even 2.0."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


@pytest.mark.parametrize("n", [-1, -2, -100])
def test_factorial_recursive_rejects_negative_values(n):
    """factorial_recursive() raises ValueError for negative inputs."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_string():
    """factorial_recursive() raises ValueError for a string input."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


def test_factorial_recursive_rejects_none():
    """factorial_recursive() raises ValueError for None."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(None)


# --------------------------------------------------------------------------- #
# __main__ block behaviour (isolated with mocks)
# --------------------------------------------------------------------------- #
def test_main_block_prints_factorial(monkeypatch, capsys):
    """Running the module as __main__ reads input and prints the factorial."""
    import runpy

    monkeypatch.setattr("builtins.input", lambda *a, **k: "5")
    runpy.run_module("factorial", run_name="__main__")
    captured = capsys.readouterr()
    assert "120" in captured.out


def test_main_block_defaults_to_zero_on_empty_input(monkeypatch, capsys):
    """Empty input in the __main__ block falls back to 0, printing 0! == 1."""
    import runpy

    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    runpy.run_module("factorial", run_name="__main__")
    captured = capsys.readouterr()
    assert "factorial0 is 1" in captured.out


def test_doctests_pass():
    """The module's embedded doctests all pass."""
    import doctest

    import factorial as factorial_module

    results = doctest.testmod(factorial_module, verbose=False)
    assert results.failed == 0
