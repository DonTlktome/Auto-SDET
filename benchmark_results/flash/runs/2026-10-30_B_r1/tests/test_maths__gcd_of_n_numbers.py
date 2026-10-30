"""Tests for maths/gcd_of_n_numbers.py"""

import importlib
from collections import Counter

import pytest

from maths.gcd_of_n_numbers import get_factors, get_greatest_common_divisor


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
        ],
    )
    def test_positive_numbers(self, number, expected):
        """Happy path: prime factorization of positive integers."""
        assert get_factors(number) == expected

    def test_one_returns_counter_with_one(self):
        """Edge case: 1 is handled specially and returns Counter({1: 1})."""
        assert get_factors(1) == Counter({1: 1})

    def test_prime_number_is_itself(self):
        """Edge case: a prime number only has itself as a factor."""
        assert get_factors(97) == Counter({97: 1})

    def test_power_of_two(self):
        """Boundary: a pure power of two yields a single factor with exponent."""
        assert get_factors(1024) == Counter({2: 10})

    def test_accumulates_into_provided_counter(self):
        """The optional `factors` argument is mutated and returned."""
        existing = Counter({7: 1})
        result = get_factors(45, factors=existing)
        assert result is existing
        assert result == Counter({7: 1, 3: 2, 5: 1})

    def test_custom_starting_factor(self):
        """Passing a starting factor skips smaller candidate divisors."""
        # 45 = 3 * 3 * 5; starting at 3 gives the same result
        assert get_factors(45, factor=3) == Counter({3: 2, 5: 1})

    @pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, "10", None])
    def test_invalid_inputs_raise_type_error(self, bad_value):
        """Edge case: non-positive or non-integer inputs raise TypeError."""
        with pytest.raises(TypeError, match="number must be integer and greater than zero"):
            get_factors(bad_value)


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
        """Boundary: coprime numbers have gcd 1."""
        assert get_greatest_common_divisor(17, 19, 23) == 1

    def test_one_among_many_returns_one(self):
        """Boundary: presence of 1 forces gcd to be 1."""
        assert get_greatest_common_divisor(1, 100, 250) == 1

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

    def test_no_arguments_raises_value_error(self):
        """Edge case: calling with no numbers raises (unpacking fails)."""
        with pytest.raises(ValueError):
            get_greatest_common_divisor()

    def test_uses_get_factors(self, monkeypatch):
        """Isolation: verify get_factors is invoked once per input number."""
        calls = []

        def fake_get_factors(number, factors=None, factor=2):
            calls.append(number)
            return Counter({2: 1})

        # Resolve the real module object instead of relying on a dotted string,
        # which would require the module to be importable under that exact name.
        module = importlib.import_module(get_greatest_common_divisor.__module__)
        monkeypatch.setattr(module, "get_factors", fake_get_factors)

        result = get_greatest_common_divisor(4, 8, 16)
        assert calls == [4, 8, 16]
        assert result == 2
