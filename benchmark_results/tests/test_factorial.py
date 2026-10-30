"""
Pytest test suite for the factorial module.

Covers:
- factorial (iterative)
- factorial_recursive (recursive)

Tests include happy paths, edge cases, boundary values, and error conditions.
"""

import math
import pytest

from maths.factorial import factorial, factorial_recursive


# ---------------------------------------------------------------------------
# factorial (iterative)
# ---------------------------------------------------------------------------

class TestFactorialIterative:
    """Tests for the iterative `factorial` function."""

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (4, 24),
            (5, 120),
            (6, 720),
            (10, 3_628_800),
        ],
    )
    def test_factorial_small_non_negative_integers(self, n, expected):
        """factorial should return correct values for small non-negative integers."""
        assert factorial(n) == expected

    @pytest.mark.parametrize("n", list(range(20)))
    def test_factorial_matches_math_library(self, n):
        """factorial should match math.factorial for integers 0..19."""
        assert factorial(n) == math.factorial(n)

    def test_factorial_large_number(self):
        """factorial should handle a moderately large integer correctly."""
        # 12! = 479001600
        assert factorial(12) == 479_001_600

    # -- Error cases --------------------------------------------------------

    @pytest.mark.parametrize("bad_input", [-1, -2, -100])
    def test_factorial_negative_raises_valueerror(self, bad_input):
        """factorial should raise ValueError for negative integers."""
        with pytest.raises(ValueError, match="not defined for negative values"):
            factorial(bad_input)

    @pytest.mark.parametrize("bad_input", [0.1, 1.5])
    def test_factorial_non_integer_float_raises_valueerror(self, bad_input):
        """factorial should raise ValueError for non-integral float values."""
        with pytest.raises(ValueError, match="only accepts integral values"):
            factorial(bad_input)

    def test_factorial_integral_float_raises_typeerror(self):
        """factorial should raise TypeError for a float that represents an integer (e.g. 3.0)."""
        with pytest.raises(TypeError):
            factorial(3.0)

    def test_factorial_string_raises_typeerror(self):
        """factorial should raise TypeError (or ValueError) for a string input."""
        # The implementation does `number != int(number)` which will raise
        # TypeError for a string because int() cannot convert it.
        with pytest.raises((TypeError, ValueError)):
            factorial("abc")

    def test_factorial_none_raises_typeerror(self):
        """factorial should raise TypeError for None input."""
        with pytest.raises(TypeError):
            factorial(None)


# ---------------------------------------------------------------------------
# factorial_recursive
# ---------------------------------------------------------------------------

class TestFactorialRecursive:
    """Tests for the recursive `factorial_recursive` function."""

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 6),
            (4, 24),
            (5, 120),
            (6, 720),
            (10, 3_628_800),
        ],
    )
    def test_factorial_recursive_small_non_negative_integers(self, n, expected):
        """factorial_recursive should return correct values for small non-negative integers."""
        assert factorial_recursive(n) == expected

    @pytest.mark.parametrize("n", list(range(20)))
    def test_factorial_recursive_matches_math_library(self, n):
        """factorial_recursive should match math.factorial for integers 0..19."""
        assert factorial_recursive(n) == math.factorial(n)

    def test_factorial_recursive_large_number(self):
        """factorial_recursive should handle a moderately large integer correctly."""
        assert factorial_recursive(12) == 479_001_600

    # -- Error cases --------------------------------------------------------

    @pytest.mark.parametrize("bad_input", [-1, -2, -100])
    def test_factorial_recursive_negative_raises_valueerror(self, bad_input):
        """factorial_recursive should raise ValueError for negative integers."""
        with pytest.raises(ValueError, match="not defined for negative values"):
            factorial_recursive(bad_input)

    @pytest.mark.parametrize("bad_input", [0.1, 1.5, 3.0])
    def test_factorial_recursive_non_integer_float_raises_valueerror(self, bad_input):
        """factorial_recursive should raise ValueError for non-integral float values."""
        with pytest.raises(ValueError, match="only accepts integral values"):
            factorial_recursive(bad_input)

    def test_factorial_recursive_string_raises_valueerror(self):
        """factorial_recursive should raise ValueError for a string input."""
        with pytest.raises(ValueError, match="only accepts integral values"):
            factorial_recursive("abc")

    def test_factorial_recursive_none_raises_valueerror(self):
        """factorial_recursive should raise ValueError for None input."""
        with pytest.raises(ValueError, match="only accepts integral values"):
            factorial_recursive(None)


# ---------------------------------------------------------------------------
# Cross-comparison: both implementations should agree
# ---------------------------------------------------------------------------

class TestConsistency:
    """Ensure both implementations produce identical results."""

    @pytest.mark.parametrize("n", list(range(30)))
    def test_both_implementations_agree(self, n):
        """factorial and factorial_recursive must return the same value."""
        assert factorial(n) == factorial_recursive(n)
