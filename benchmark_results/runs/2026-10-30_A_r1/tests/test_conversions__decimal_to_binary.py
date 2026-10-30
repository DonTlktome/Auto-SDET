import pytest

from decimal_to_binary import (
    decimal_to_binary_iterative,
    decimal_to_binary_recursive,
    decimal_to_binary_recursive_helper,
)


class TestDecimalToBinaryIterative:
    """Tests for decimal_to_binary_iterative."""

    @pytest.mark.parametrize(
        ("num", "expected"),
        [
            (0, "0b0"),
            (1, "0b1"),
            (2, "0b10"),
            (7, "0b111"),
            (35, "0b100011"),
            (1000, "0b1111101000"),
        ],
    )
    def test_positive_integers(self, num, expected):
        """Positive integers should convert to their binary representation."""
        assert decimal_to_binary_iterative(num) == expected

    @pytest.mark.parametrize(
        ("num", "expected"),
        [
            (-1, "-0b1"),
            (-2, "-0b10"),
            (-7, "-0b111"),
            (-35, "-0b100011"),
        ],
    )
    def test_negative_integers(self, num, expected):
        """Negative integers should produce a '-0b' prefixed binary string."""
        assert decimal_to_binary_iterative(num) == expected

    def test_zero_edge_case(self):
        """Zero is a boundary value and should return '0b0'."""
        assert decimal_to_binary_iterative(0) == "0b0"

    def test_float_raises_type_error(self):
        """Floats should raise a TypeError."""
        with pytest.raises(TypeError):
            decimal_to_binary_iterative(16.16)

    def test_string_raises_type_error(self):
        """Strings should raise a TypeError."""
        with pytest.raises(TypeError):
            decimal_to_binary_iterative("0xfffff")


class TestDecimalToBinaryRecursiveHelper:
    """Tests for decimal_to_binary_recursive_helper."""

    @pytest.mark.parametrize(
        ("decimal", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (2, "10"),
            (72, "1001000"),
            (1000, "1111101000"),
        ],
    )
    def test_positive_integers(self, decimal, expected):
        """Positive integers should convert to a binary string without prefix."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    def test_string_input_is_coerced(self):
        """Numeric strings should be coerced to int and converted."""
        assert decimal_to_binary_recursive_helper("72") == "1001000"

    def test_non_numeric_string_raises_value_error(self):
        """Non-numeric strings should raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive_helper("number")

    def test_zero_and_one_boundary(self):
        """Zero and one are the recursion exit cases."""
        assert decimal_to_binary_recursive_helper(0) == "0"
        assert decimal_to_binary_recursive_helper(1) == "1"


class TestDecimalToBinaryRecursive:
    """Tests for decimal_to_binary_recursive."""

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (0, "0b0"),
            (1, "0b1"),
            (40, "0b101000"),
            (1000, "0b1111101000"),
        ],
    )
    def test_positive_integers(self, number, expected):
        """Positive integers should produce a '0b' prefixed binary string."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize(
        ("number", "expected"),
        [
            (-1, "-0b1"),
            (-40, "-0b101000"),
            (-1000, "-0b1111101000"),
        ],
    )
    def test_negative_integers(self, number, expected):
        """Negative integers should produce a '-0b' prefixed binary string."""
        assert decimal_to_binary_recursive(number) == expected

    def test_string_input_is_accepted(self):
        """Numeric strings should be accepted and converted."""
        assert decimal_to_binary_recursive("40") == "0b101000"

    def test_whitespace_is_stripped(self):
        """Leading/trailing whitespace should be stripped before conversion."""
        assert decimal_to_binary_recursive("  40  ") == "0b101000"

    def test_float_raises_value_error(self):
        """Floats should raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive(40.8)

    def test_non_numeric_string_raises_value_error(self):
        """Non-numeric strings should raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive("forty")

    def test_empty_string_raises_value_error(self):
        """An empty string should raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive("")

    def test_whitespace_only_raises_value_error(self):
        """A whitespace-only string should raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive("   ")