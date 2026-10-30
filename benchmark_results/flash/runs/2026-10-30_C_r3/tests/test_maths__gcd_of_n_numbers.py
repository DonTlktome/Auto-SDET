"""Tests for maths/gcd_of_n_numbers.py"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcd_of_n_numbers import get_factors, get_greatest_common_divisor


class TestGetFactors:
    """Tests for the recursive get_factors helper."""

    @pytest.mark.parametrize(
        "number, expected",
        [
            (45, Counter({3: 2, 5: 1})),
            (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
            (23, Counter({23: 1})),
            (2, Counter({2: 1})),
            (4, Counter({2: 2})),
            (100, Counter({2: 2, 5: 2})),
            (1024, Counter({2: 10})),
        ],
    )
    def test_prime_factorization(self, number, expected):
        """get_factors returns the correct prime factorization for positive ints."""
        assert get_factors(number) == expected

    def test_one_returns_counter_with_one(self):
        """The number 1 is a special case returning Counter({1: 1})."""
        assert get_factors(1) == Counter({1: 1})

    def test_large_prime(self):
        """A large prime number should factorize to itself with exponent 1."""
        prime = 7919
        # get_factors recurses once per candidate divisor, so factoring a prime
        # near 8000 needs thousands of nested calls; CPython's default limit of
        # 1000 is far too small.  Temporarily raise it for this test only.
        original_limit = sys.getrecursionlimit()
        required_limit = 4 * prime
        sys.setrecursionlimit(max(original_limit, required_limit))
        try:
            assert get_factors(prime) == Counter({prime: 1})
        finally:
            sys.setrecursionlimit(original_limit)

    @pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, "10", None])
    def test_invalid_inputs_raise_type_error(self, bad_value):
        """Non-positive or non-integer inputs raise TypeError."""
        with pytest.raises(TypeError, match="number must be integer and greater than zero"):
            get_factors(bad_value)

    def test_accumulates_into_provided_counter(self):
        """Passing an existing Counter accumulates factors into it."""
        existing = Counter({7: 1})
        result = get_factors(45, existing)
        assert result[7] == 1
        assert result[3] == 2
        assert result[5] == 1

    def test_custom_starting_factor(self):
        """Starting from a factor greater than 2 skips smaller primes."""
        # 45 = 3 * 3 * 5; starting at 3 yields the same factorization.
        assert get_factors(45, None, 3) == Counter({3: 2, 5: 1})


class TestGetGreatestCommonDivisor:
    """Tests for get_greatest_common_divisor."""

    @pytest.mark.parametrize(
        "numbers, expected",
        [
            ((18, 45), 9),
            ((23, 37), 1),
            ((2520, 8350), 10),
            ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1),
            ((12, 18, 24), 6),
            ((7, 7, 7), 7),
            ((100, 100), 100),
            ((2, 4, 8, 16), 2),
            ((36, 60, 84), 12),
        ],
    )
    def test_gcd_values(self, numbers, expected):
        """GCD of various integer tuples matches the expected value."""
        assert get_greatest_common_divisor(*numbers) == expected

    def test_single_number_returns_itself(self):
        """GCD of a single number is the number itself."""
        assert get_greatest_common_divisor(42) == 42

    def test_single_one_returns_one(self):
        """GCD of a single 1 is 1."""
        assert get_greatest_common_divisor(1) == 1

    def test_coprime_numbers(self):
        """Consecutive integers are always coprime."""
        assert get_greatest_common_divisor(17, 18) == 1

    def test_large_numbers(self):
        """GCD works for large integers."""
        assert get_greatest_common_divisor(123456, 789012) == 12

    @pytest.mark.parametrize(
        "numbers",
        [
            (-10, 20),
            (0, 5),
            (1.5, 2),
            ("1", 2, 3),
            (10, None),
        ],
    )
    def test_invalid_inputs_raise_exception(self, numbers):
        """Invalid inputs raise a generic Exception with the documented message."""
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(*numbers)

    def test_no_arguments_raises_value_error(self):
        """Calling with no numbers raises ValueError from tuple unpacking."""
        with pytest.raises(ValueError):
            get_greatest_common_divisor()
