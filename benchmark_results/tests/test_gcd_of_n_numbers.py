"""
Pytest test suite for gcd_of_n_numbers.py

Covers:
- get_factors: positive cases, edge cases (1, primes, large numbers),
  invalid inputs (0, negative, float), and internal recursion behavior.
- get_greatest_common_divisor: positive cases (2 numbers, many numbers,
  coprime, one-number edge case), invalid inputs (negative, float, string),
  and boundary conditions.
"""

from collections import Counter
from unittest.mock import patch, MagicMock

import pytest
import sys

sys.setrecursionlimit(2000)

from gcd_of_n_numbers import get_factors, get_greatest_common_divisor


# ---------------------------------------------------------------------------
# get_factors tests
# ---------------------------------------------------------------------------

class TestGetFactors:
    """Tests for the recursive get_factors function."""

    # -- Positive / happy-path tests ----------------------------------------

    @pytest.mark.parametrize(
        "number, expected",
        [
            (1, Counter({1: 1})),
            (2, Counter({2: 1})),
            (3, Counter({3: 1})),
            (4, Counter({2: 2})),
            (6, Counter({2: 1, 3: 1})),
            (8, Counter({2: 3})),
            (9, Counter({3: 2})),
            (45, Counter({3: 2, 5: 1})),
            (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
            (23, Counter({23: 1})),
            (97, Counter({97: 1})),  # prime
            (100, Counter({2: 2, 5: 2})),
        ],
    )
    def test_positive_cases(self, number, expected):
        """get_factors should return the correct prime-factor Counter for
        various positive integers."""
        assert get_factors(number) == expected

    def test_large_prime(self):
        """get_factors should handle a larger prime number correctly."""
        # 997 is prime
        result = get_factors(997)
        assert result == Counter({997: 1})

    def test_large_composite(self):
        """get_factors should factor a larger composite number correctly."""
        # 2^5 * 3^3 = 32 * 27 = 864
        result = get_factors(864)
        assert result == Counter({2: 5, 3: 3})

    # -- Edge-case / boundary tests -----------------------------------------

    def test_number_one_returns_special_counter(self):
        """get_factors(1) returns Counter({1: 1}) — a special case."""
        result = get_factors(1)
        assert result == Counter({1: 1})
        assert isinstance(result, Counter)

    def test_power_of_two(self):
        """get_factors should correctly factor powers of two."""
        # 2^10 = 1024
        result = get_factors(1024)
        assert result == Counter({2: 10})

    def test_product_of_two_large_primes(self):
        """get_factors should factor a product of two distinct primes."""
        # 17 * 19 = 323
        result = get_factors(323)
        assert result == Counter({17: 1, 19: 1})

    # -- Invalid-input tests ------------------------------------------------

    @pytest.mark.parametrize(
        "bad_input",
        [
            0,
            -1,
            -100,
        ],
    )
    def test_zero_or_negative_raises_typeerror(self, bad_input):
        """get_factors should raise TypeError for zero or negative integers."""
        with pytest.raises(TypeError, match="number must be integer and greater than zero"):
            get_factors(bad_input)

    @pytest.mark.parametrize(
        "bad_input",
        [
            1.5,
            2.0,
            3.14,
        ],
    )
    def test_float_raises_typeerror(self, bad_input):
        """get_factors should raise TypeError for float inputs."""
        with pytest.raises(TypeError, match="number must be integer and greater than zero"):
            get_factors(bad_input)

    def test_string_raises_typeerror(self):
        """get_factors should raise TypeError for a string input."""
        with pytest.raises(TypeError, match="number must be integer and greater than zero"):
            get_factors("45")

    # -- Internal recursion / factors-accumulation tests --------------------

    def test_explicit_factor_start(self):
        """Providing an explicit starting factor should still produce the
        correct Counter."""
        # 45 = 3*3*5; starting factor=3 should skip 2
        result = get_factors(45, factor=3)
        assert result == Counter({3: 2, 5: 1})

    def test_passing_existing_factors_counter(self):
        """Passing an existing Counter as `factors` should accumulate into it."""
        existing = Counter({2: 1})
        # 6 = 2 * 3; with existing {2:1}, result should be {2:2, 3:1}
        result = get_factors(6, factors=existing)
        assert result == Counter({2: 2, 3: 1})

    def test_passing_both_factors_and_factor(self):
        """Passing both an existing Counter and a starting factor should work."""
        existing = Counter({3: 1})
        # 9 = 3*3; starting factor=3, existing {3:1} → {3:3}
        result = get_factors(9, factors=existing, factor=3)
        assert result == Counter({3: 3})


# ---------------------------------------------------------------------------
# get_greatest_common_divisor tests
# ---------------------------------------------------------------------------

class TestGetGreatestCommonDivisor:
    """Tests for get_greatest_common_divisor(*numbers)."""

    # -- Positive / happy-path tests ----------------------------------------

    @pytest.mark.parametrize(
        "numbers, expected",
        [
            ((18, 45), 9),
            ((23, 37), 1),
            ((2520, 8350), 10),
            ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1),
            ((12, 18, 24), 6),
            ((100, 75, 50), 25),
            ((7, 14, 21, 28), 7),
            ((64, 128, 256), 64),
            ((17, 34), 17),
        ],
    )
    def test_positive_cases(self, numbers, expected):
        """get_greatest_common_divisor should return the correct GCD for
        various sets of positive integers."""
        assert get_greatest_common_divisor(*numbers) == expected

    def test_single_number(self):
        """GCD of a single number should be the number itself."""
        assert get_greatest_common_divisor(42) == 42

    def test_single_number_one(self):
        """GCD of a single 1 should be 1."""
        assert get_greatest_common_divisor(1) == 1

    def test_coprime_pair(self):
        """GCD of two coprime numbers should be 1."""
        assert get_greatest_common_divisor(14, 15) == 1

    def test_all_equal_numbers(self):
        """GCD of identical numbers should be that number."""
        assert get_greatest_common_divisor(12, 12, 12, 12) == 12

    def test_large_numbers(self):
        """GCD should work correctly with large numbers."""
        # GCD(123456, 789012) = 12
        assert get_greatest_common_divisor(123456, 789012) == 12

    # -- Edge-case / boundary tests -----------------------------------------

    def test_two_ones(self):
        """GCD(1, 1) should be 1."""
        assert get_greatest_common_divisor(1, 1) == 1

    def test_one_and_another(self):
        """GCD(1, N) should be 1 for any N > 0."""
        assert get_greatest_common_divisor(1, 999) == 1

    def test_prime_numbers(self):
        """GCD of two distinct primes should be 1."""
        assert get_greatest_common_divisor(13, 17) == 1

    def test_one_number_is_multiple_of_other(self):
        """GCD(a, b) where b is a multiple of a should be a."""
        assert get_greatest_common_divisor(7, 49) == 7

    # -- Invalid-input tests ------------------------------------------------

    @pytest.mark.parametrize(
        "bad_args",
        [
            (-10, 20),
            (10, -20),
            (0, 5),
            (5, 0),
            (-1, -1),
        ],
    )
    def test_zero_or_negative_raises_exception(self, bad_args):
        """get_greatest_common_divisor should raise Exception for any
        non-positive integer argument."""
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(*bad_args)

    @pytest.mark.parametrize(
        "bad_args",
        [
            (1.5, 2),
            (3, 4.0),
            (2.5, 3.5),
        ],
    )
    def test_float_raises_exception(self, bad_args):
        """get_greatest_common_divisor should raise Exception for float args."""
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(*bad_args)

    @pytest.mark.parametrize(
        "bad_args",
        [
            ("1", 2, 3),
            (1, "2"),
            ("abc",),
        ],
    )
    def test_string_raises_exception(self, bad_args):
        """get_greatest_common_divisor should raise Exception for string args."""
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(*bad_args)

    def test_mixed_valid_and_invalid(self):
        """If any argument is invalid, Exception should be raised."""
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(10, -5, 3)

    # -- Interaction / integration-style tests ------------------------------

    def test_calls_get_factors_for_each_number(self):
        """get_greatest_common_divisor should delegate to get_factors for
        each input number."""
        with patch(
            "gcd_of_n_numbers.get_factors",
            side_effect=[Counter({2: 1, 3: 1}), Counter({2: 1, 5: 1})],
        ) as mock_get_factors:
            result = get_greatest_common_divisor(6, 10)
            # 6 → {2:1, 3:1}, 10 → {2:1, 5:1}, intersection → {2:1} → 2
            assert result == 2
            assert mock_get_factors.call_count == 2

    def test_no_numbers_provided(self):
        """Calling get_greatest_common_divisor with no arguments should
        raise a ValueError (from unpacking)."""
        with pytest.raises(ValueError, match="not enough values to unpack"):
            get_greatest_common_divisor()
