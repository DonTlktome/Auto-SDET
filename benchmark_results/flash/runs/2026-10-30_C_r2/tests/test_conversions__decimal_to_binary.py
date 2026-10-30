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
        """Positive integers are converted to a '0b'-prefixed binary string."""
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
        """Zero returns '0b0' rather than an empty binary body."""
        assert decimal_to_binary_iterative(0) == "0b0"

    def test_matches_builtin_bin(self):
        """Output matches Python's built-in bin() for a range of values."""
        for value in list(range(-50, 51)):
            assert decimal_to_binary_iterative(value) == bin(value)

    def test_float_raises_type_error(self):
        """Floats raise TypeError with the documented message."""
        with pytest.raises(TypeError) as exc_info:
            decimal_to_binary_iterative(16.16)
        assert "'float' object cannot be interpreted as an integer" in str(
            exc_info.value
        )

    def test_string_raises_type_error(self):
        """Strings raise TypeError with the documented message."""
        with pytest.raises(TypeError) as exc_info:
            decimal_to_binary_iterative("0xfffff")
        assert "'str' object cannot be interpreted as an integer" in str(
            exc_info.value
        )

    @pytest.mark.parametrize("bad_value", [0.0, -1.5, "", "12", "abc"])
    def test_invalid_types_raise_type_error(self, bad_value):
        """Any float or str input raises TypeError."""
        with pytest.raises(TypeError):
            decimal_to_binary_iterative(bad_value)

    def test_bool_input_behaves_like_int(self):
        """Booleans are ints in Python, so True -> '0b1' and False -> '0b0'."""
        assert decimal_to_binary_iterative(True) == "0b1"
        assert decimal_to_binary_iterative(False) == "0b0"

    def test_large_integer(self):
        """A large integer is converted correctly."""
        assert decimal_to_binary_iterative(2**20) == "0b" + "1" + "0" * 20


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
        """Integers return the binary digits without any prefix."""
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
        """A non-numeric string raises ValueError from int()."""
        with pytest.raises(ValueError) as exc_info:
            decimal_to_binary_recursive_helper("number")
        assert "invalid literal for int()" in str(exc_info.value)

    def test_empty_string_raises_value_error(self):
        """An empty string raises ValueError from int()."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive_helper("")

    def test_float_truncates_via_int(self):
        """Floats are truncated by int() before conversion."""
        assert decimal_to_binary_recursive_helper(2.9) == "10"

    def test_matches_builtin_bin_without_prefix(self):
        """Output equals bin(n)[2:] for positive values."""
        for value in range(0, 64):
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
            (-40, "-0b101000"),
            (-1000, "-0b1111101000"),
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
            ("\t-7\n", "-0b111"),
            ("0", "0b0"),
        ],
    )
    def test_string_inputs_are_stripped(self, number, expected):
        """String inputs are stripped of surrounding whitespace before parsing."""
        assert decimal_to_binary_recursive(number) == expected

    @pytest.mark.parametrize("bad_value", [40.8, "forty", "4.5", "1e3", "0x1f"])
    def test_non_integer_inputs_raise_value_error(self, bad_value):
        """Non-integer inputs raise ValueError with the documented message."""
        with pytest.raises(ValueError) as exc_info:
            decimal_to_binary_recursive(bad_value)
        assert "Input value is not an integer" in str(exc_info.value)

    @pytest.mark.parametrize("empty_value", ["", "   ", "\t\n"])
    def test_empty_input_raises_value_error(self, empty_value):
        """Blank or whitespace-only input raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            decimal_to_binary_recursive(empty_value)
        assert "No input value was provided" in str(exc_info.value)

    def test_float_input_raises_value_error(self):
        """A float input is rejected rather than truncated."""
        with pytest.raises(ValueError):
            decimal_to_binary_recursive(40.8)

    def test_matches_builtin_bin(self):
        """Output matches bin() for a range of positive and negative values."""
        for value in list(range(-30, 31)):
            assert decimal_to_binary_recursive(value) == bin(value)

    def test_large_integer(self):
        """A large integer is converted correctly."""
        assert decimal_to_binary_recursive(2**16) == "0b1" + "0" * 16

    def test_consistency_with_iterative(self):
        """Recursive and iterative implementations agree on the same inputs."""
        for value in list(range(-25, 26)):
            assert decimal_to_binary_recursive(value) == decimal_to_binary_iterative(
                value
            )