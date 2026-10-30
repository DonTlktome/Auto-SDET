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
        """Convert non-negative integers to their binary representation."""
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
        """Convert negative integers to a prefixed negative binary string."""
        assert decimal_to_binary_iterative(num) == expected

    def test_float_raises_type_error(self):
        """A float input must raise a TypeError."""
        with pytest.raises(TypeError):
            decimal_to_binary_iterative(16.16)

    def test_string_raises_type_error(self):
        """A string input must raise a TypeError."""
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
        """Convert positive integers to a binary string without prefix."""
        assert decimal_to_binary_recursive_helper(decimal) == expected

    def test_numeric_string_input(self):
        """A numeric string is coerced to int and converted."""
        assert decimal_to_binary_recursive_helper("72") == "1001000"

    def test_invalid_string_raises_value_error(self):
        """A non-numeric string must raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive_helper("number")


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
        """Convert positive integers to a prefixed binary string."""
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
        """Convert negative integers to a negative-prefixed binary string."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize(
        "number",
        [
            "40",
            " 40 ",
            "-40",
            " -40 ",
        ],
    )
    def test_string_inputs_with_whitespace(self, number):
        """String inputs (with optional whitespace/sign) are accepted."""
        expected = "-0b101000" if number.strip().startswith("-") else "0b101000"
        assert decimal_to_binary_recursive(number) == expected

    def test_float_raises_value_error(self):
        """A float input must raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive(40.8)

    def test_non_numeric_string_raises_value_error(self):
        """A non-numeric string must raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive("forty")

    def test_empty_string_raises_value_error(self):
        """An empty string must raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive("")

    def test_whitespace_only_raises_value_error(self):
        """A whitespace-only string must raise a ValueError."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive("   ")
