"""Tests for maths/perfect_number.py."""

import pytest

from maths.perfect_number import perfect


class TestPerfectHappyPath:
    """Positive test cases for the perfect() function."""

    @pytest.mark.parametrize(
        "number",
        [6, 28, 496, 8128],
    )
    def test_known_perfect_numbers(self, number):
        """Known perfect numbers must be reported as perfect."""
        assert perfect(number) is True

    @pytest.mark.parametrize(
        "number",
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 27, 29, 100],
    )
    def test_known_non_perfect_numbers(self, number):
        """Known non-perfect positive integers must be reported as not perfect."""
        assert perfect(number) is False

    def test_large_perfect_number(self):
        """A large perfect number (33550336) must be detected as perfect."""
        assert perfect(33550336) is True

    def test_large_number_just_above_perfect(self):
        """A number just above a large perfect number must not be perfect."""
        assert perfect(33550337) is False


class TestPerfectEdgeCases:
    """Boundary and edge cases for the perfect() function."""

    @pytest.mark.parametrize("number", [0, -1, -6, -28, -100])
    def test_zero_and_negative_numbers(self, number):
        """Zero and negative numbers are never perfect numbers."""
        assert perfect(number) is False

    def test_one_is_not_perfect(self):
        """1 has no proper divisors, so it is not a perfect number."""
        assert perfect(1) is False

    def test_two_is_not_perfect(self):
        """2's only proper divisor is 1, so it is not perfect."""
        assert perfect(2) is False

    def test_return_type_is_bool(self):
        """The function must return a real bool, not a truthy/falsy value."""
        assert isinstance(perfect(6), bool)
        assert isinstance(perfect(7), bool)


class TestPerfectInvalidInput:
    """Invalid input handling for the perfect() function."""

    @pytest.mark.parametrize(
        "number",
        ["123", "Hello", "", "6", 12.34, 0.0, -1.5, None, [6], (6,), {"n": 6}],
    )
    def test_non_integer_input_raises_value_error(self, number):
        """Non-integer inputs must raise ValueError with the documented message."""
        with pytest.raises(ValueError) as exc_info:
            perfect(number)
        assert str(exc_info.value) == "number must be an integer"

    def test_bool_input_is_accepted_as_int_subclass(self):
        """bool is a subclass of int, so it is accepted (True == 1 -> not perfect)."""
        assert perfect(True) is False
        assert perfect(False) is False

    def test_float_that_is_integral_still_raises(self):
        """A float like 6.0 is not an int instance and must raise ValueError."""
        with pytest.raises(ValueError, match="number must be an integer"):
            perfect(6.0)