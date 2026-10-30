"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorial import factorial, factorial_recursive  # noqa: E402


# ---------------------------------------------------------------------------
# factorial() — happy path
# ---------------------------------------------------------------------------


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


def test_factorial_returns_int_type():
    """factorial() returns an int, not a float or other numeric type."""
    result = factorial(7)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


# ---------------------------------------------------------------------------
# factorial() — edge cases / errors
# ---------------------------------------------------------------------------


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1."""
    assert factorial(1) == 1


def test_factorial_negative_raises_value_error():
    """Negative input raises ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(-1)


@pytest.mark.parametrize("number", [-1, -5, -1000])
def test_factorial_various_negatives_raise(number):
    """Any negative integer raises ValueError."""
    with pytest.raises(ValueError):
        factorial(number)


@pytest.mark.parametrize("number", [0.1, 1.5, 2.999, -0.5, 3.0])
def test_factorial_non_integral_raises(number):
    """Non-integral floats raise ValueError; integral floats hit a TypeError in range().

    Note: the source only checks ``number != int(number)``, so an integral
    float such as 3.0 slips past the guard and then fails inside
    ``range()`` with a TypeError rather than being accepted.
    """
    if number == int(number):
        with pytest.raises(TypeError):
            factorial(number)
    else:
        with pytest.raises(ValueError, match="only accepts integral values"):
            factorial(number)


def test_factorial_integral_float_raises_type_error():
    """An integral float (5.0) passes the != int() check but range() raises TypeError."""
    with pytest.raises(TypeError):
        factorial(5.0)


def test_factorial_large_value():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


# ---------------------------------------------------------------------------
# factorial_recursive() — happy path
# ---------------------------------------------------------------------------


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


def test_factorial_recursive_agrees_with_iterative():
    """Both implementations produce identical results."""
    assert all(factorial(i) == factorial_recursive(i) for i in range(25))


def test_factorial_recursive_returns_int_type():
    """factorial_recursive() returns an int."""
    assert isinstance(factorial_recursive(6), int)


# ---------------------------------------------------------------------------
# factorial_recursive() — edge cases / errors
# ---------------------------------------------------------------------------


def test_factorial_recursive_zero_is_one():
    """Boundary: base case 0! == 1."""
    assert factorial_recursive(0) == 1


def test_factorial_recursive_one_is_one():
    """Boundary: base case 1! == 1."""
    assert factorial_recursive(1) == 1


def test_factorial_recursive_negative_raises_value_error():
    """Negative input raises ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(-1)


@pytest.mark.parametrize("n", [-1, -2, -100])
def test_factorial_recursive_various_negatives_raise(n):
    """Any negative integer raises ValueError."""
    with pytest.raises(ValueError):
        factorial_recursive(n)


@pytest.mark.parametrize("n", [0.1, 1.5, 2.5, -0.5])
def test_factorial_recursive_non_int_raises(n):
    """Non-int types raise ValueError, including integral floats like 3.0."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_integral_float():
    """Unlike factorial(), factorial_recursive() rejects 5.0 via isinstance."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


@pytest.mark.parametrize("bad_value", ["5", None, [5], (5,)])
def test_factorial_recursive_rejects_non_numeric_types(bad_value):
    """Non-numeric types raise ValueError rather than TypeError."""
    with pytest.raises(ValueError):
        factorial_recursive(bad_value)


def test_factorial_recursive_recursion_depth_limit():
    """Very large inputs exceed Python's recursion limit (documented behaviour)."""
    with pytest.raises(RecursionError):
        factorial_recursive(5000)


# ---------------------------------------------------------------------------
# Isolation / mocking
# ---------------------------------------------------------------------------


def test_factorial_uses_no_external_dependencies():
    """factorial() is pure: patching builtins.open must not affect it."""
    with patch("builtins.open", side_effect=AssertionError("no I/O expected")):
        assert factorial(5) == 120


def test_factorial_recursive_uses_no_external_dependencies():
    """factorial_recursive() is pure: patching builtins.open must not affect it."""
    with patch("builtins.open", side_effect=AssertionError("no I/O expected")):
        assert factorial_recursive(5) == 120


def test_factorial_recursive_calls_itself():
    """factorial_recursive() recurses for n > 1 (verified via call counting)."""
    original = factorial_recursive
    calls = []

    def counting(n):
        calls.append(n)
        return original(n)

    with patch.object(sys.modules["factorial"], "factorial_recursive", counting):
        # The module-level name is patched, so the recursive lookup inside the
        # function body resolves to the counting wrapper.
        assert counting(4) == 24
    # Recursion descends 4 -> 3 -> 2 -> 1 before hitting the base case.
    assert calls == [4, 3, 2, 1]
