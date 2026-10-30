"""Tests for maths/factorial.py"""

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
    """factorial(n) returns the correct value for small non-negative integers."""
    assert factorial(n) == expected


def test_factorial_matches_math_factorial():
    """factorial agrees with math.factorial for a range of inputs."""
    for i in range(20):
        assert factorial(i) == math.factorial(i)


def test_factorial_zero_is_one():
    """Boundary: factorial(0) must be 1 (empty product)."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: factorial(1) must be 1."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial handles larger inputs correctly (20! is a known constant)."""
    assert factorial(20) == 2432902008176640000


def test_factorial_returns_int_type():
    """factorial always returns an int."""
    assert isinstance(factorial(5), int)


@pytest.mark.parametrize("value", [0.1, 1.5, 2.999, -0.5, 3.0000001])
def test_factorial_rejects_non_integral_floats(value):
    """Non-integral floats raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(value)


@pytest.mark.parametrize("value", [1.0, 2.0, 5.0, 0.0])
def test_factorial_rejects_integral_floats(value):
    """Integral-valued floats pass the integrality check but are not usable.

    The implementation only verifies ``number != int(number)`` and then feeds
    the value straight into ``range(1, number + 1)``, which requires an int and
    therefore raises ``TypeError`` for floats such as ``1.0`` or ``0.0``.
    """
    with pytest.raises(TypeError):
        factorial(value)


@pytest.mark.parametrize("value", [-1, -2, -100])
def test_factorial_rejects_negative_values(value):
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(value)


def test_factorial_negative_float_raises_integral_error_first():
    """A negative non-integral float fails the integral check first."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(-1.5)


def test_factorial_rejects_non_numeric_string():
    """Passing a non-numeric string raises (TypeError from int()/comparison)."""
    with pytest.raises((TypeError, ValueError)):
        factorial("abc")


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
    """factorial_recursive(n) returns the correct value for small inputs."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial():
    """factorial_recursive agrees with math.factorial for a range of inputs."""
    for i in range(20):
        assert factorial_recursive(i) == math.factorial(i)


def test_factorial_recursive_zero_is_one():
    """Boundary: factorial_recursive(0) must be 1."""
    assert factorial_recursive(0) == 1


def test_factorial_recursive_one_is_one():
    """Boundary: factorial_recursive(1) must be 1 (base case)."""
    assert factorial_recursive(1) == 1


def test_factorial_recursive_agrees_with_iterative():
    """Both implementations produce identical results."""
    for i in range(15):
        assert factorial_recursive(i) == factorial(i)


def test_factorial_recursive_returns_int_type():
    """factorial_recursive always returns an int."""
    assert isinstance(factorial_recursive(5), int)


@pytest.mark.parametrize("value", [0.1, 1.5, 2.999, -0.5])
def test_factorial_recursive_rejects_non_int_types(value):
    """Non-int types (including integral floats) raise ValueError."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(value)


def test_factorial_recursive_rejects_integral_float():
    """Unlike the iterative version, factorial_recursive rejects 5.0."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


@pytest.mark.parametrize("value", [-1, -2, -100])
def test_factorial_recursive_rejects_negative_values(value):
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(value)


def test_factorial_recursive_rejects_string():
    """A string input raises ValueError (fails the isinstance check)."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


def test_factorial_recursive_rejects_bool():
    """Booleans are ints in Python, so True behaves like 1 (documented quirk)."""
    assert factorial_recursive(True) == 1
    assert factorial_recursive(False) == 1
