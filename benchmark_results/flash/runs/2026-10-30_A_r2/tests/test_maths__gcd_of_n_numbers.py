"""Tests for maths/gcd_of_n_numbers.py"""

from collections import Counter

import pytest

from maths.gcd_of_n_numbers import get_factors, get_greatest_common_divisor


class TestGetFactors:
    """Tests for the recursive get_factors function."""

    @pytest.mark.parametrize(
        "number, expected",
        [
            (2, Counter({2: 1})),
            (3, Counter({3: 1})),
            (4, Counter({2: 2})),
            (45, Counter({3: 2, 5: 1})),
            (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
            (23, Counter({23: 1})),
            (100, Counter({2: 2, 5: 2})),
            (1024, Counter({2: 10})),
        ],
    )
    def test_get_factors_happy_path(self, number, expected):
        """get_factors returns the correct prime factorization Counter."""
        assert get_factors(number) == expected

    def test_get_factors_one(self):
        """get_factors(1) returns Counter({1: 1}) as a special case."""
        assert get_factors(1) == Counter({1: 1})

    def test_get_factors_returns_counter_type(self):
        """get_factors always returns a Counter instance."""
        assert isinstance(get_factors(12), Counter)

    def test_get_factors_accepts_prefilled_counter(self):
        """A pre-populated Counter is mutated/extended rather than replaced."""
        initial = Counter({2: 1})
        result = get_factors(4, factors=initial)
        assert result[2] == 3

    def test_get_factors_custom_starting_factor(self):
        """Starting from a factor equal to the number yields that number as factor."""
        assert get_factors(7, factor=7) == Counter({7: 1})

    @pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, "5", None])
    def test_get_factors_invalid_input_raises_type_error(self, bad_value):
        """Non-positive or non-integer inputs raise TypeError."""
        with pytest.raises(TypeError):
            get_factors(bad_value)

    def test_get_factors_error_message(self):
        """The TypeError carries the documented message."""
        with pytest.raises(TypeError, match="number must be integer and greater than zero"):
            get_factors(0)


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
            ((100, 10), 10),
            ((2, 4, 8, 16), 2),
            ((1, 1), 1),
        ],
    )
    def test_gcd_happy_path(self, numbers, expected):
        """GCD of several positive integers is computed correctly."""
        assert get_greatest_common_divisor(*numbers) == expected

    def test_gcd_single_number(self):
        """GCD of a single number is the number itself."""
        assert get_greatest_common_divisor(42) == 42

    def test_gcd_single_number_one(self):
        """GCD of the single number 1 is 1."""
        assert get_greatest_common_divisor(1) == 1

    def test_gcd_coprime_numbers(self):
        """Coprime numbers have a GCD of 1."""
        assert get_greatest_common_divisor(17, 19, 23) == 1

    def test_gcd_result_divides_all_inputs(self):
        """The returned GCD divides every input number."""
        numbers = (36, 60, 84)
        result = get_greatest_common_divisor(*numbers)
        assert all(n % result == 0 for n in numbers)

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
    def test_gcd_invalid_input_raises_exception(self, numbers):
        """Invalid inputs raise a generic Exception with the documented message."""
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(*numbers)

    def test_gcd_no_arguments_raises(self):
        """Calling with no arguments raises (unpacking of empty map fails)."""
        with pytest.raises(Exception):
            get_greatest_common_divisor()