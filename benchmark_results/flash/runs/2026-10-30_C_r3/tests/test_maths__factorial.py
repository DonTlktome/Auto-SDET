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
def test_factorial_known_values(n, expected):
    """factorial() returns the correct value for small non-negative integers."""
    assert factorial(n) == expected


@pytest.mark.parametrize("n", list(range(0, 20)))
def test_factorial_matches_math_factorial(n):
    """factorial() agrees with math.factorial for n in [0, 20)."""
    assert factorial(n) == math.factorial(n)


def test_factorial_zero_is_one():
    """Boundary: factorial(0) must be 1 (empty product)."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: factorial(1) must be 1."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial() handles a moderately large input correctly."""
    assert factorial(20) == 2432902008176640000


@pytest.mark.parametrize("bad", [0.1, 1.5, 2.999, -0.5, 3.0000001])
def test_factorial_rejects_non_integral_floats(bad):
    """factorial() raises ValueError for non-integral float inputs."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(bad)


@pytest.mark.parametrize("integral_float", [0.0, 1.0, 5.0, 10.0])
def test_factorial_integral_floats_fail_in_range(integral_float):
    """Integral floats pass the value check but then break range() with TypeError."""
    with pytest.raises(TypeError):
        factorial(integral_float)


@pytest.mark.parametrize("negative", [-1, -2, -100])
def test_factorial_rejects_negative_values(negative):
    """factorial() raises ValueError for negative integers."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(negative)


def test_factorial_negative_float_raises_integral_error_first():
    """A negative non-integral float fails the integral check before the sign check."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(-1.5)


def test_factorial_rejects_non_numeric_string():
    """factorial() raises ValueError for non-numeric string inputs."""
    with pytest.raises(ValueError):
        factorial("5")


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
def test_factorial_recursive_known_values(n, expected):
    """factorial_recursive() returns the correct value for small non-negative ints."""
    assert factorial_recursive(n) == expected


@pytest.mark.parametrize("n", list(range(0, 20)))
def test_factorial_recursive_matches_math_factorial(n):
    """factorial_recursive() agrees with math.factorial for n in [0, 20)."""
    assert factorial_recursive(n) == math.factorial(n)


@pytest.mark.parametrize("n", list(range(0, 20)))
def test_iterative_and_recursive_agree(n):
    """Both implementations produce identical results for the same input."""
    assert factorial(n) == factorial_recursive(n)


def test_factorial_recursive_zero_is_one():
    """Boundary: factorial_recursive(0) must be 1 (base case)."""
    assert factorial_recursive(0) == 1


def test_factorial_recursive_one_is_one():
    """Boundary: factorial_recursive(1) must be 1 (base case)."""
    assert factorial_recursive(1) == 1


def test_factorial_recursive_large_value():
    """factorial_recursive() handles a moderately large input correctly."""
    assert factorial_recursive(20) == 2432902008176640000


@pytest.mark.parametrize("bad", [0.1, 1.5, 2.999, -0.5, 3.0000001])
def test_factorial_recursive_rejects_non_integral_floats(bad):
    """factorial_recursive() raises ValueError for non-int inputs, including integral floats."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(bad)


@pytest.mark.parametrize("integral_float", [0.0, 1.0, 5.0])
def test_factorial_recursive_rejects_integral_floats(integral_float):
    """Unlike factorial(), factorial_recursive() uses isinstance and rejects all floats."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(integral_float)


@pytest.mark.parametrize("negative", [-1, -2, -100])
def test_factorial_recursive_rejects_negative_values(negative):
    """factorial_recursive() raises ValueError for negative integers."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(negative)


def test_factorial_recursive_rejects_string():
    """factorial_recursive() raises ValueError for non-int types such as str."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


def test_factorial_recursive_rejects_bool_like_non_int():
    """factorial_recursive() rejects None as a non-integral value."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(None)
