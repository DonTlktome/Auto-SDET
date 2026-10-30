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
            (1000, "0b1111101000"),
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

    def test_zero_edge_case(self):
        """Zero is a boundary case handled before the main loop."""
        assert decimal_to_binary_iterative(0) == "0b0"

    def test_large_power_of_two(self):
        """A large power of two yields a single 1 followed by zeros."""
        assert decimal_to_binary_iterative(2**20) == "0b1" + "0" * 20

    def test_matches_builtin_bin(self):
        """Output matches Python's builtin bin() for a range of values."""
        for value in range(-50, 51):
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
            (40, "101000"),
            (1000, "1111101000"),
        ],
    )
    def test_integer_inputs(self, decimal, expected):
        """Integers are converted to a bare binary string (no prefix)."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    @pytest.mark.parametrize(
        "decimal, expected",
        [
            ("0", "0"),
            ("1", "1"),
            ("72", "1001000"),
            ("1000", "1111101000"),
        ],
    )
    def test_numeric_string_inputs(self, decimal, expected):
        """Numeric strings are coerced via int() and converted."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    def test_non_numeric_string_raises_value_error(self):
        """A non-numeric string raises ValueError from int()."""
        with pytest.raises(ValueError, match="invalid literal for int"):
            decimal_to_binary_recursive_helper("number")

    def test_empty_string_raises_value_error(self):
        """An empty string raises ValueError from int()."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive_helper("")

    def test_float_truncates(self):
        """Floats are truncated by int() before conversion."""
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
            (2, "0b10"),
            (40, "0b101000"),
            (1000, "0b1111101000"),
        ],
    )
    def test_positive_integers(self, number, expected):
        """Positive integers get a '0b' prefix."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize(
        "number, expected",
        [
            (-1, "-0b1"),
            (-2, "-0b10"),
            (-40, "-0b101000"),
        ],
    )
    def test_negative_integers(self, number, expected):
        """Negative integers get a '-0b' prefix."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize(
        "number, expected",
        [
            ("40", "0b101000"),
            ("-40", "-0b101000"),
            ("  40  ", "0b101000"),
            ("0", "0b0"),
        ],
    )
    def test_string_inputs(self, number, expected):
        """Numeric strings (including whitespace and sign) are accepted."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize("number", [40.8, "forty", "4.5", "0x1f", "1e3"])
    def test_invalid_inputs_raise_value_error(self, number):
        """Non-integer inputs raise ValueError with the documented message."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive(number)

    @pytest.mark.parametrize("number", ["", "   ", "\t\n"])
    def test_empty_input_raises_value_error(self, number):
        """Blank or whitespace-only input raises ValueError."""
        with pytest.raises(ValueError, match="No input value was provided"):
            decimal_to_binary_recursive(number)

    def test_zero_edge_case(self):
        """Zero is the smallest boundary value and yields '0b0'."""
        assert decimal_to_binary_recursive(0) == "0b0"

    def test_matches_builtin_bin(self):
        """Output matches Python's builtin bin() across a range of values."""
        for value in range(-100, 101):
            assert decimal_to_binary_recursive(value) == bin(value)

    def test_consistency_with_iterative(self):
        """Recursive and iterative implementations agree on valid inputs."""
        for value in range(-64, 65):
            assert decimal_to_binary_recursive(value) == decimal_to_binary_iterative(
                value
            )

    def test_large_value(self):
        """A large integer is converted correctly."""
        assert decimal_to_binary_recursive(2**30) == "0b1" + "0" * 30