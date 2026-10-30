"""Tests for conversions/decimal_to_binary.py."""

import pytest

from conversions.decimal_to_binary import (
    decimal_to_binary_iterative,
    decimal_to_binary_recursive,
    decimal_to_binary_recursive_helper,
)


class TestDecimalToBinaryIterative:
    """Tests for decimal_to_binary_iterative."""

    @pytest.mark.parametrize(
        "num, expected",
        [
            (0, "0b0"),
            (1, "0b1"),
            (2, "0b10"),
            (7, "0b111"),
            (35, "0b100011"),
            (255, "0b11111111"),
            (1024, "0b10000000000"),
        ],
    )
    def test_positive_integers(self, num, expected):
        """Positive integers are converted with a '0b' prefix."""
        assert decimal_to_binary_iterative(num) == expected

    @pytest.mark.parametrize(
        "num, expected",
        [
            (-1, "-0b1"),
            (-2, "-0b10"),
            (-7, "-0b111"),
            (-35, "-0b100011"),
        ],
    )
    def test_negative_integers(self, num, expected):
        """Negative integers are converted with a '-0b' prefix."""
        assert decimal_to_binary_iterative(num) == expected

    def test_zero_is_special_cased(self):
        """Zero returns '0b0' and never enters the loop."""
        assert decimal_to_binary_iterative(0) == "0b0"

    def test_matches_builtin_bin(self):
        """Output matches Python's builtin bin() for a range of values."""
        for value in range(-50, 51):
            assert decimal_to_binary_iterative(value) == bin(value)

    def test_float_raises_type_error(self):
        """Floats raise TypeError with the documented message."""
        with pytest.raises(
            TypeError, match="'float' object cannot be interpreted as an integer"
        ):
            decimal_to_binary_iterative(16.16)

    def test_string_raises_type_error(self):
        """Strings raise TypeError with the documented message."""
        with pytest.raises(
            TypeError, match="'str' object cannot be interpreted as an integer"
        ):
            decimal_to_binary_iterative("0xfffff")

    def test_bool_is_accepted_as_int(self):
        """Booleans are ints in Python, so they convert without error."""
        assert decimal_to_binary_iterative(True) == "0b1"
        assert decimal_to_binary_iterative(False) == "0b0"


class TestDecimalToBinaryRecursiveHelper:
    """Tests for decimal_to_binary_recursive_helper."""

    @pytest.mark.parametrize(
        "decimal, expected",
        [
            (0, "0"),
            (1, "1"),
            (2, "10"),
            (3, "11"),
            (1000, "1111101000"),
            (255, "11111111"),
        ],
    )
    def test_positive_integers(self, decimal, expected):
        """Positive integers convert to binary without any prefix."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    @pytest.mark.parametrize(
        "decimal, expected",
        [
            ("72", "1001000"),
            ("0", "0"),
            ("1", "1"),
            (" 42 ", "101010"),
        ],
    )
    def test_numeric_strings(self, decimal, expected):
        """Numeric strings are coerced via int() before conversion."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    def test_non_numeric_string_raises_value_error(self):
        """Non-numeric strings raise ValueError from int()."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive_helper("number")

    def test_float_truncates_toward_zero(self):
        """Floats are truncated by int() rather than rejected."""
        assert decimal_to_binary_recursive_helper(2.9) == "10"

    def test_matches_builtin_bin_without_prefix(self):
        """Output equals bin() output with the '0b' prefix stripped."""
        for value in range(0, 100):
            assert decimal_to_binary_recursive_helper(value) == bin(value)[2:]


class TestDecimalToBinaryRecursive:
    """Tests for decimal_to_binary_recursive."""

    @pytest.mark.parametrize(
        "number, expected",
        [
            (0, "0b0"),
            (1, "0b1"),
            (40, "0b101000"),
            (1000, "0b1111101000"),
            ("40", "0b101000"),
            ("  40  ", "0b101000"),
        ],
    )
    def test_positive_inputs(self, number, expected):
        """Positive ints and numeric strings get a '0b' prefix."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize(
        "number, expected",
        [
            (-1, "-0b1"),
            (-40, "-0b101000"),
            ("-40", "-0b101000"),
            (" -40 ", "-0b101000"),
        ],
    )
    def test_negative_inputs(self, number, expected):
        """Negative ints and numeric strings get a '-0b' prefix."""
        assert decimal_to_binary_recursive(number) == expected

    def test_zero(self):
        """Zero is handled by the helper's base case."""
        assert decimal_to_binary_recursive(0) == "0b0"

    @pytest.mark.parametrize("number", [40.8, "forty", "4.5", "abc", "0x1f"])
    def test_invalid_inputs_raise_value_error(self, number):
        """Non-integer inputs raise ValueError with the documented message."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive(number)

    @pytest.mark.parametrize("number", ["", "   ", "\t\n"])
    def test_empty_input_raises_value_error(self, number):
        """Blank or whitespace-only input raises a dedicated ValueError."""
        with pytest.raises(ValueError, match="No input value was provided"):
            decimal_to_binary_recursive(number)

    def test_matches_builtin_bin(self):
        """Output matches Python's builtin bin() across a range of values."""
        for value in range(-100, 101):
            assert decimal_to_binary_recursive(value) == bin(value)

    def test_leading_zeros_are_accepted(self):
        """Leading zeros in a numeric string are tolerated."""
        assert decimal_to_binary_recursive("007") == "0b111"

    def test_negative_zero_string(self):
        """'-0' is treated as zero and keeps the negative prefix."""
        assert decimal_to_binary_recursive("-0") == "-0b0"