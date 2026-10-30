"""Tests for maths/gcd_of_n_numbers.py"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcd_of_n_numbers import get_factors, get_greatest_common_divisor


class TestGetFactors:
    """Tests for the recursive get_factors function."""

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
    def test_positive_numbers(self, number, expected):
        """Happy path: prime factorization of positive integers."""
        assert get_factors(number) == expected

    def test_one_returns_counter_with_one(self):
        """Edge case: 1 has no prime factors; implementation returns Counter({1: 1})."""
        assert get_factors(1) == Counter({1: 1})

    def test_prime_number_returns_itself(self):
        """Edge case: a prime number is its own only factor."""
        assert get_factors(97) == Counter({97: 1})

    @pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, "10", None])
    def test_invalid_inputs_raise_type_error(self, bad_value):
        """Edge case: non-positive or non-integer inputs raise TypeError."""
        with pytest.raises(TypeError, match="number must be integer and greater than zero"):
            get_factors(bad_value)

    def test_accumulates_into_provided_counter(self):
        """Edge case: a pre-populated Counter is mutated and returned."""
        existing = Counter({7: 1})
        result = get_factors(45, factors=existing)
        assert result is existing
        assert result[3] == 2
        assert result[5] == 1
        assert result[7] == 1

    def test_custom_starting_factor(self):
        """Edge case: starting factor above 2 skips smaller primes."""
        # 45 = 3 * 3 * 5; starting at 3 yields the same factorization.
        assert get_factors(45, factor=3) == Counter({3: 2, 5: 1})

    def test_large_number(self):
        """Boundary: a moderately large number factorizes correctly.

        The implementation walks candidate factors one by one and recurses, so
        a large prime would need ~number recursive calls (RecursionError).
        2**20 is large in magnitude but reaches the break condition after only
        20 divisions, so it stays within Python's default recursion limit.
        """
        assert get_factors(2**20) == Counter({2: 20})

    def test_large_prime_starting_at_itself(self):
        """Boundary: a large prime is detected immediately when it is the factor."""
        assert get_factors(999983, factor=999983) == Counter({999983: 1})


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
            ((100, 10, 1000), 10),
            ((2, 4, 8, 16), 2),
            ((36, 60, 84), 12),
        ],
    )
    def test_valid_inputs(self, numbers, expected):
        """Happy path: gcd of several positive integers."""
        assert get_greatest_common_divisor(*numbers) == expected

    def test_single_number_returns_itself(self):
        """Edge case: gcd of a single number is the number itself."""
        assert get_greatest_common_divisor(42) == 42

    def test_single_one_returns_one(self):
        """Edge case: gcd of a single 1 is 1."""
        assert get_greatest_common_divisor(1) == 1

    def test_coprime_numbers_return_one(self):
        """Edge case: pairwise coprime numbers have gcd 1."""
        assert get_greatest_common_divisor(13, 17, 19, 23) == 1

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
        """Edge case: invalid inputs are wrapped into a generic Exception."""
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(*numbers)

    def test_no_arguments_raises(self):
        """Edge case: calling with no numbers raises (unpacking empty map)."""
        with pytest.raises(Exception):
            get_greatest_common_divisor()

    def test_uses_get_factors(self, monkeypatch):
        """Isolation: verify get_factors is invoked once per input number."""
        calls = []
        real_get_factors = get_factors

        def spy(number, *args, **kwargs):
            # get_factors recurses through the module-level name, so only the
            # outermost calls (a single positional argument) count as one
            # invocation per input number.
            if not args and not kwargs:
                calls.append(number)
            return real_get_factors(number, *args, **kwargs)

        monkeypatch.setattr("gcd_of_n_numbers.get_factors", spy)
        assert get_greatest_common_divisor(18, 45) == 9
        assert calls == [18, 45]

    def test_propagates_type_error_as_exception(self, monkeypatch):
        """Isolation: a TypeError from get_factors is re-raised as Exception."""

        def boom(*args, **kwargs):
            raise TypeError("boom")

        monkeypatch.setattr("gcd_of_n_numbers.get_factors", boom)
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(4, 8)
