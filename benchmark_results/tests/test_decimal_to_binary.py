"""
Pytest test suite for the decimal_to_binary module.

Tests cover:
    - decimal_to_binary_iterative
    - decimal_to_binary_recursive_helper
    - decimal_to_binary_recursive
"""

import pytest

from decimal_to_binary import (
    decimal_to_binary_iterative,
    decimal_to_binary_recursive,
    decimal_to_binary_recursive_helper,
)


# ---------------------------------------------------------------------------
# decimal_to_binary_iterative
# ---------------------------------------------------------------------------

class TestDecimalToBinaryIterative:
    """Tests for decimal_to_binary_iterative."""

    @pytest.mark.parametrize(
        ("decimal", "expected"),
        [
            (0, "0b0"),
            (1, "0b1"),
            (2, "0b10"),
            (7, "0b111"),
            (8, "0b1000"),
            (35, "0b100011"),
            (255, "0b11111111"),
            (1024, "0b10000000000"),
        ],
    )
    def test_positive_integers(self, decimal, expected):
        """Positive integers should return the correct '0b'-prefixed binary string."""
        assert decimal_to_binary_iterative(decimal) == expected

    @pytest.mark.parametrize(
        ("decimal", "expected"),
        [
            (-1, "-0b1"),
            (-2, "-0b10"),
            (-7, "-0b111"),
            (-35, "-0b100011"),
            (-255, "-0b11111111"),
        ],
    )
    def test_negative_integers(self, decimal, expected):
        """Negative integers should return the correct '-0b'-prefixed binary string."""
        assert decimal_to_binary_iterative(decimal) == expected

    def test_large_integer(self):
        """A large positive integer should be converted correctly."""
        result = decimal_to_binary_iterative(10**6)
        assert result == "0b" + bin(10**6)[2:]

    def test_large_negative_integer(self):
        """A large negative integer should be converted correctly."""
        result = decimal_to_binary_iterative(-(10**6))
        assert result == "-0b" + bin(10**6)[2:]

    def test_float_raises_type_error(self):
        """Passing a float should raise TypeError."""
        with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
            decimal_to_binary_iterative(16.16)

    def test_string_raises_type_error(self):
        """Passing a string should raise TypeError."""
        with pytest.raises(TypeError, match="'str' object cannot be interpreted as an integer"):
            decimal_to_binary_iterative("0xfffff")

    def test_bool_input(self):
        """Booleans are subclasses of int and should be accepted."""
        assert decimal_to_binary_iterative(True) == "0b1"
        assert decimal_to_binary_iterative(False) == "0b0"


# ---------------------------------------------------------------------------
# decimal_to_binary_recursive_helper
# ---------------------------------------------------------------------------

class TestDecimalToBinaryRecursiveHelper:
    """Tests for decimal_to_binary_recursive_helper."""

    @pytest.mark.parametrize(
        ("decimal", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (2, "10"),
            (7, "111"),
            (1000, "1111101000"),
            (255, "11111111"),
        ],
    )
    def test_positive_integers(self, decimal, expected):
        """Positive integers should return the binary string without prefix."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    @pytest.mark.parametrize(
        ("decimal_str", "expected"),
        [
            ("72", "1001000"),
            ("0", "0"),
            ("1", "1"),
        ],
    )
    def test_string_inputs(self, decimal_str, expected):
        """String representations of integers should be parsed and converted."""
        assert decimal_to_binary_recursive_helper(decimal_str) == expected

    def test_invalid_string_raises_value_error(self):
        """A non-numeric string should raise ValueError."""
        with pytest.raises(ValueError, match=r"invalid literal for int\(\) with base 10: 'number'"):
            decimal_to_binary_recursive_helper("number")

    def test_large_integer(self):
        """A large integer should be handled correctly."""
        result = decimal_to_binary_recursive_helper(10**6)
        assert result == bin(10**6)[2:]


# ---------------------------------------------------------------------------
# decimal_to_binary_recursive
# ---------------------------------------------------------------------------

class TestDecimalToBinaryRecursive:
    """Tests for decimal_to_binary_recursive."""

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (0, "0b0"),
            (1, "0b1"),
            (40, "0b101000"),
            (255, "0b11111111"),
            (1000, "0b1111101000"),
        ],
    )
    def test_positive_integers(self, number, expected):
        """Positive integers should return the correct '0b'-prefixed binary string."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (-1, "-0b1"),
            (-40, "-0b101000"),
            (-255, "-0b11111111"),
        ],
    )
    def test_negative_integers(self, number, expected):
        """Negative integers should return the correct '-0b'-prefixed binary string."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            ("0", "0b0"),
            ("40", "0b101000"),
            ("-40", "-0b101000"),
            ("  42  ", "0b101010"),
            ("  -7  ", "-0b111"),
        ],
    )
    def test_string_inputs(self, number, expected):
        """String representations of integers (with optional whitespace) should work."""
        assert decimal_to_binary_recursive(number) == expected

    def test_float_raises_value_error(self):
        """A float should raise ValueError because it is not an integer."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive(40.8)

    def test_non_numeric_string_raises_value_error(self):
        """A non-numeric string should raise ValueError."""
        with pytest.raises(ValueError, match="Input value is not an integer"):
            decimal_to_binary_recursive("forty")

    def test_empty_string_raises_value_error(self):
        """An empty string should raise ValueError."""
        with pytest.raises(ValueError, match="No input value was provided"):
            decimal_to_binary_recursive("")

    def test_whitespace_only_string_raises_value_error(self):
        """A whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="No input value was provided"):
            decimal_to_binary_recursive("   ")

    def test_large_integer(self):
        """A large integer should be converted correctly."""
        result = decimal_to_binary_recursive(10**6)
        assert result == "0b" + bin(10**6)[2:]

    def test_large_negative_integer(self):
        """A large negative integer should be converted correctly."""
        result = decimal_to_binary_recursive(-(10**6))
        assert result == "-0b" + bin(10**6)[2:]

    def test_zero_boundary(self):
        """Zero should be handled correctly (edge case)."""
        assert decimal_to_binary_recursive(0) == "0b0"
        assert decimal_to_binary_recursive("0") == "0b0"
