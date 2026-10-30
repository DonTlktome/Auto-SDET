"""
Tests for the gcd_of_n_numbers module.

Covers the public functions:
- get_factors
- get_greatest_common_divisor
"""

import pytest
from collections import Counter

from maths.gcd_of_n_numbers import get_factors, get_greatest_common_divisor


# ---------------------------------------------------------------------------
# get_factors
# ---------------------------------------------------------------------------

class TestGetFactors:
    """Tests for the get_factors function."""

    @pytest.mark.parametrize(
        "number, expected",
        [
            (1, Counter({1: 1})),
            (2, Counter({2: 1})),
            (23, Counter({23: 1})),
            (45, Counter({3: 2, 5: 1})),
            (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
            (8, Counter({2: 3})),
            (97, Counter({97: 1})),
        ],
    )
    def test_get_factors_positive(self, number, expected):
        """Prime factorization should return the correct Counter."""
        assert get_factors(number) == expected

    @pytest.mark.parametrize("number", [0, -1, -100])
    def test_get_factors_non_positive_raises(self, number):
        """Non-positive integers should raise TypeError."""
        with pytest.raises(TypeError):
            get_factors(number)

    @pytest.mark.parametrize("number", [1.5, 2.0, "3", None])
    def test_get_factors_non_integer_raises(self, number):
        """Non-integer inputs should raise TypeError."""
        with pytest.raises(TypeError):
            get_factors(number)

    def test_get_factors_with_initial_factors(self):
        """Passing an existing Counter should accumulate into it."""
        factors = Counter({2: 1})
        result = get_factors(8, factors)
        assert result == Counter({2: 4})

    def test_get_factors_with_custom_factor(self):
        """A custom starting factor should be respected."""
        # Starting at factor 3 skips the factor 2 of 12.
        result = get_factors(12, factor=3)
        assert result == Counter({3: 1, 4: 1})


# ---------------------------------------------------------------------------
# get_greatest_common_divisor
# ---------------------------------------------------------------------------

class TestGetGreatestCommonDivisor:
    """Tests for the get_greatest_common_divisor function."""

    @pytest.mark.parametrize(
        "numbers, expected",
        [
            ((18, 45), 9),
            ((23, 37), 1),
            ((2520, 8350), 10),
            ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1),
            ((12, 18, 24), 6),
            ((100, 100, 100), 100),
            ((7,), 7),
            ((48, 64, 80), 16),
        ],
    )
    def test_gcd_positive(self, numbers, expected):
        """GCD of valid positive integers should be computed correctly."""
        assert get_greatest_common_divisor(*numbers) == expected

    @pytest.mark.parametrize(
        "numbers",
        [
            (-10, 20),
            (0, 5),
            (1.5, 2),
            ("1", 2, 3),
            (None, 2),
        ],
    )
    def test_gcd_invalid_input_raises(self, numbers):
        """Invalid inputs should raise Exception."""
        with pytest.raises(Exception):
            get_greatest_common_divisor(*numbers)

    def test_gcd_single_number(self):
        """GCD of a single number is the number itself."""
        assert get_greatest_common_divisor(42) == 42

    def test_gcd_coprime_numbers(self):
        """GCD of coprime numbers should be 1."""
        assert get_greatest_common_divisor(17, 19) == 1