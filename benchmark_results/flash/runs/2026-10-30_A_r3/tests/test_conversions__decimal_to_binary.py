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
            (-35, "-0b100011"),
            (-1024, "-0b10000000000"),
        ],
    )
    def test_negative_integers(self, num, expected):
        """Negative integers are converted with a '-0b' prefix."""
        assert decimal_to_binary_iterative(num) == expected

    def test_zero_is_boundary(self):
        """Zero is the loop boundary and must return '0b0' without a sign."""
        assert decimal_to_binary_iterative(0) == "0b0"

    def test_matches_builtin_bin(self):
        """Output must match Python's builtin bin() for a range of values."""
        for value in list(range(-50, 51)):
            assert decimal_to_binary_iterative(value) == bin(value)

    def test_float_raises_type_error(self):
        """Floats are rejected with a TypeError."""
        with pytest.raises(TypeError, match="'float' object cannot be interpreted"):
            decimal_to_binary_iterative(16.16)

    def test_string_raises_type_error(self):
        """Strings are rejected with a TypeError."""
        with pytest.raises(TypeError, match="'str' object cannot be interpreted"):
            decimal_to_binary_iterative("0xfffff")

    def test_bool_is_accepted_as_int(self):
        """Booleans are int subclasses and are converted accordingly."""
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
            (1024, "10000000000"),
        ],
    )
    def test_integer_inputs(self, decimal, expected):
        """Integers are converted to a bare binary string (no prefix)."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    @pytest.mark.parametrize(
        "decimal, expected",
        [
            ("72", "1001000"),
            ("0", "0"),
            ("1", "1"),
            ("255", "11111111"),
        ],
    )
    def test_numeric_string_inputs(self, decimal, expected):
        """Numeric strings are coerced via int() and converted."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    def test_non_numeric_string_raises_value_error(self):
        """Non-numeric strings raise ValueError from int()."""
        with pytest.raises(ValueError, match="invalid literal for int"):
            decimal_to_binary_recursive_helper("number")

    def test_empty_string_raises_value_error(self):
        """An empty string raises ValueError from int()."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive_helper("")

    def test_matches_builtin_bin_without_prefix(self):
        """Output equals bin(n)[2:] for a range of non-negative values."""
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
        """Positive integers and numeric strings get a '0b' prefix."""
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
        """Negative integers and numeric strings get a '-0b' prefix."""
        assert decimal_to_binary_recursive(number) == expected

    def test_zero_boundary(self):
        """Zero returns '0b0' with no sign."""
        assert decimal_to_binary_recursive(0) == "0b0"

    def test_float_raises_value_error(self):
        """Floats are rejected with a ValueError."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive(40.8)

    def test_non_numeric_string_raises_value_error(self):
        """Non-numeric strings are rejected with a ValueError."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive("forty")

    @pytest.mark.parametrize("empty", ["", "   ", "\t", "\n"])
    def test_empty_input_raises_value_error(self, empty):
        """Blank or whitespace-only input raises a dedicated ValueError."""
        with pytest.raises(ValueError, match="No input value was provided"):
            decimal_to_binary_recursive(empty)

    def test_matches_builtin_bin(self):
        """Output matches bin() for a range of positive and negative values."""
        for value in list(range(-64, 65)):
            assert decimal_to_binary_recursive(value) == bin(value)

    def test_consistency_with_iterative(self):
        """Both implementations agree on the same inputs."""
        for value in list(range(-100, 101)):
            assert decimal_to_binary_recursive(value) == decimal_to_binary_iterative(
                value
            )