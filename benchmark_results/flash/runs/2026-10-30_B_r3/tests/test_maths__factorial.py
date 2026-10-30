"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math

import pytest

from maths.factorial import factorial, factorial_recursive


# ---------------------------------------------------------------------------
# factorial (iterative)
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
def test_factorial_happy_path(n, expected):
    """factorial(n) returns the mathematically correct value for small n."""
    assert factorial(n) == expected


def test_factorial_matches_math_factorial():
    """factorial agrees with math.factorial for a range of inputs."""
    for i in range(20):
        assert factorial(i) == math.factorial(i)


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial handles a moderately large input without overflow."""
    assert factorial(20) == 2432902008176640000


def test_factorial_integral_float_raises_type_error():
    """An integral float (e.g. 5.0) passes the integrality check but is not

    usable by the internal ``range()`` call, so a TypeError is raised.
    """
    with pytest.raises(TypeError):
        factorial(5.0)


@pytest.mark.parametrize("bad", [0.1, 1.5, -2.5, 3.0001])
def test_factorial_rejects_non_integral_values(bad):
    """Non-integral numeric inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(bad)


@pytest.mark.parametrize("bad", [-1, -5, -100])
def test_factorial_rejects_negative_values(bad):
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(bad)


def test_factorial_negative_float_raises_integral_error_first():
    """A negative non-integral value fails the integral check before the sign check."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(-1.5)


def test_factorial_returns_int_type():
    """The return value is always a Python int."""
    assert isinstance(factorial(7), int)


# ---------------------------------------------------------------------------
# factorial_recursive
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
def test_factorial_recursive_happy_path(n, expected):
    """factorial_recursive(n) returns the mathematically correct value for small n."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial():
    """factorial_recursive agrees with math.factorial for a range of inputs."""
    for i in range(20):
        assert factorial_recursive(i) == math.factorial(i)


def test_factorial_recursive_matches_iterative():
    """Both implementations produce identical results for the same inputs."""
    for i in range(15):
        assert factorial_recursive(i) == factorial(i)


def test_factorial_recursive_zero_is_one():
    """Boundary: 0! is defined as 1 in the recursive implementation."""
    assert factorial_recursive(0) == 1


def test_factorial_recursive_one_is_one():
    """Boundary: 1! is 1 (base case of the recursion)."""
    assert factorial_recursive(1) == 1


def test_factorial_recursive_large_value():
    """factorial_recursive handles a moderately large input."""
    assert factorial_recursive(20) == 2432902008176640000


@pytest.mark.parametrize("bad", [0.1, 1.5, -2.5, 3.0001])
def test_factorial_recursive_rejects_non_integral_values(bad):
    """Non-int inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(bad)


def test_factorial_recursive_rejects_integral_float():
    """Unlike the iterative version, an integral float is rejected (isinstance check)."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


@pytest.mark.parametrize("bad", [-1, -5, -100])
def test_factorial_recursive_rejects_negative_values(bad):
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(bad)


def test_factorial_recursive_returns_int_type():
    """The return value is always a Python int."""
    assert isinstance(factorial_recursive(7), int)
