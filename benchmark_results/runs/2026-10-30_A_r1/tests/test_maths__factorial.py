# AUTO-SDET: Test generation FAILED after 3 retries
# Last error: 

"""
Pytest test suite for the factorial module.

Covers the iterative ``factorial`` and recursive ``factorial_recursive``
functions, including happy paths, edge cases, and error handling.
"""

import math

import pytest

from maths.factorial import factorial, factorial_recursive


# ---------------------------------------------------------------------------
# factorial (iterative)
# ---------------------------------------------------------------------------

class TestFactorial:
    """Tests for the iterative ``factorial`` function."""

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (6, 720),
            (10, 3628800),
        ],
    )
    def test_factorial_positive_values(self, number, expected):
        """Verify factorial returns correct results for non-negative integers."""
        assert factorial(number) == expected

    @pytest.mark.parametrize("number", list(range(20)))
    def test_factorial_matches_math_factorial(self, number):
        """Verify factorial agrees with math.factorial for a range of inputs."""
        assert factorial(number) == math.factorial(number)

    def test_factorial_large_value(self):
        """Verify factorial handles a moderately large input correctly."""
        assert factorial(20) == math.factorial(20)

    @pytest.mark.parametrize("number", [0.1, 1.5, 2.0 + 1e-9, "3", None])
    def test_factorial_non_integral_raises(self, number):
        """Verify factorial raises ValueError for non-integral inputs."""
        with pytest.raises(ValueError, match="only accepts integral values"):
            factorial(number)

    def test_factorial_negative_raises(self):
        """Verify factorial raises ValueError for negative inputs."""
        with pytest.raises(ValueError, match="not defined for negative values"):
            factorial(-1)

    def test_factorial_negative_boundary(self):
        """Verify factorial raises for the smallest negative integer boundary."""
        with pytest.raises(ValueError, match="not defined for negative values"):
            factorial(-1)


# ---------------------------------------------------------------------------
# factorial_recursive
# ---------------------------------------------------------------------------

class TestFactorialRecursive:
    """Tests for the recursive ``factorial_recursive`` function."""

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (6, 720),
            (10, 3628800),
        ],
    )
    def test_factorial_recursive_positive_values(self, number, expected):
        """Verify factorial_recursive returns correct results for non-negative ints."""
        assert factorial_recursive(number) == expected

    @pytest.mark.parametrize("number", list(range(20)))
    def test_factorial_recursive_matches_math_factorial(self, number):
        """Verify factorial_recursive agrees with math.factorial for a range."""
        assert factorial_recursive(number) == math.factorial(number)

    def test_factorial_recursive_large_value(self):
        """Verify factorial_recursive handles a moderately large input correctly."""
        assert factorial_recursive(20) == math.factorial(20)

    @pytest.mark.parametrize("number", [0.1, 1.5, "3", None, True])
    def test_factorial_recursive_non_integral_raises(self, number):
        """Verify factorial_recursive raises ValueError for non-int inputs."""
        with pytest.raises(ValueError, match="only accepts integral values"):
            factorial_recursive(number)

    def test_factorial_recursive_negative_raises(self):
        """Verify factorial_recursive raises ValueError for negative inputs."""
        with pytest.raises(ValueError, match="not defined for negative values"):
            factorial_recursive(-1)

    def test_factorial_recursive_negative_boundary(self):
        """Verify factorial_recursive raises for the smallest negative boundary."""
        with pytest.raises(ValueError, match="not defined for negative values"):
            factorial_recursive(-1)