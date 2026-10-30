"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import maths.factorial as factorial_module  # noqa: E402
from maths.factorial import factorial, factorial_recursive  # noqa: E402

MODULE_FILE = Path(factorial_module.__file__)


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
    """factorial() agrees with math.factorial across a range of inputs."""
    assert all(factorial(i) == math.factorial(i) for i in range(30))


def test_factorial_returns_int_type():
    """factorial() returns an int, not a float or other numeric type."""
    result = factorial(7)
    assert isinstance(result, int)
    assert result == 5040


# --------------------------------------------------------------------------- #
# factorial() — edge / boundary cases
# --------------------------------------------------------------------------- #
def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1 (smallest positive input)."""
    assert factorial(1) == 1


def test_factorial_large_input():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


@pytest.mark.parametrize("number", [0.1, 1.5, 2.0000001, -0.5, 3.7])
def test_factorial_rejects_non_integral_values(number):
    """factorial() raises ValueError for non-integral numeric input."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(number)


@pytest.mark.parametrize("number", [-1, -2, -100])
def test_factorial_rejects_negative_values(number):
    """factorial() raises ValueError for negative input."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(number)


def test_factorial_integral_float_raises_type_error():
    """An exactly-integral float passes the integral check but then the
    internal ``range(1, number + 1)`` rejects the float with a TypeError."""
    with pytest.raises(TypeError):
        factorial(5.0)


def test_factorial_rejects_non_numeric():
    """factorial() rejects a non-numeric string with a ValueError because the
    ``number != int(number)`` comparison fails."""
    with pytest.raises(ValueError, match="only accepts integral values"):
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
    """factorial_recursive() returns the correct value for small inputs."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial():
    """factorial_recursive() agrees with math.factorial across a range."""
    assert all(factorial_recursive(i) == math.factorial(i) for i in range(20))


def test_factorial_recursive_matches_iterative():
    """Both implementations agree for the same inputs."""
    assert all(factorial(i) == factorial_recursive(i) for i in range(25))


# --------------------------------------------------------------------------- #
# factorial_recursive() — edge / boundary cases
# --------------------------------------------------------------------------- #
def test_factorial_recursive_zero_is_one():
    """Boundary: recursive base case for 0 returns 1."""
    assert factorial_recursive(0) == 1


def test_factorial_recursive_one_is_one():
    """Boundary: recursive base case for 1 returns 1."""
    assert factorial_recursive(1) == 1


@pytest.mark.parametrize("n", [0.1, 1.5, 2.5, -0.5])
def test_factorial_recursive_rejects_non_int(n):
    """factorial_recursive() raises ValueError for non-int input (floats)."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_integral_float():
    """Unlike factorial(), factorial_recursive() rejects even integral floats."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


@pytest.mark.parametrize("n", [-1, -2, -50])
def test_factorial_recursive_rejects_negative(n):
    """factorial_recursive() raises ValueError for negative input."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_string():
    """factorial_recursive() raises ValueError for a string input."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


def test_factorial_recursive_rejects_none():
    """factorial_recursive() raises ValueError for None input."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(None)


# --------------------------------------------------------------------------- #
# __main__ block behaviour
# --------------------------------------------------------------------------- #
def test_main_block_prints_factorial(capsys):
    """Running the module as __main__ reads input and prints the factorial."""
    with patch("builtins.input", return_value="5"):
        runpy.run_path(str(MODULE_FILE), run_name="__main__")

    captured = capsys.readouterr()
    assert "factorial5 is 120" in captured.out


def test_main_block_defaults_to_zero_on_empty_input(capsys):
    """An empty input line in the __main__ block defaults to 0."""
    with patch("builtins.input", return_value=""):
        runpy.run_path(str(MODULE_FILE), run_name="__main__")

    captured = capsys.readouterr()
    assert "factorial0 is 1" in captured.out
