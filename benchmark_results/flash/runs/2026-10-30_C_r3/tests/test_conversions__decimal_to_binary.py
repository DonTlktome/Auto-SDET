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
        """Zero returns '0b0' and never '-0b0'."""
        assert decimal_to_binary_iterative(0) == "0b0"
        assert decimal_to_binary_iterative(-0) == "0b0"

    def test_large_integer(self):
        """A large integer converts correctly (boundary of many loop iterations)."""
        assert decimal_to_binary_iterative(2**20) == "0b1" + "0" * 20

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

    def test_matches_builtin_bin(self):
        """Output matches Python's built-in bin() for a range of values."""
        for value in range(-50, 51):
            assert decimal_to_binary_iterative(value) == bin(value)


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
        """Integers convert to a bare binary string with no prefix."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    @pytest.mark.parametrize(
        "decimal, expected",
        [
            ("72", "1001000"),
            ("0", "0"),
            ("1", "1"),
            (" 8 ", "1000"),
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
        """Floats are truncated by int() rather than rejected."""
        assert decimal_to_binary_recursive_helper(2.9) == "10"

    def test_matches_builtin_bin_for_positive_range(self):
        """Output matches bin() without the prefix for positive values."""
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
            (" 40 ", "0b101000"),
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

    def test_none_raises_value_error(self):
        """None stringifies to 'None', which is not numeric."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive(None)

    def test_leading_minus_on_numeric_string_is_normalized(self):
        """lstrip('-') removes every leading dash, so '--5' is treated as -5."""
        assert decimal_to_binary_recursive("--5") == "-0b101"

    def test_non_numeric_remainder_after_minus_raises_value_error(self):
        """A non-leading '-' leaves a non-numeric remainder and raises ValueError."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive("5-")

    def test_matches_builtin_bin(self):
        """Output matches Python's built-in bin() across a range of values."""
        for value in range(-100, 101):
            assert decimal_to_binary_recursive(value) == bin(value)

    def test_consistency_with_iterative(self):
        """Both implementations agree on the same inputs."""
        for value in range(-64, 65):
            assert decimal_to_binary_recursive(value) == decimal_to_binary_iterative(
                value
            )
