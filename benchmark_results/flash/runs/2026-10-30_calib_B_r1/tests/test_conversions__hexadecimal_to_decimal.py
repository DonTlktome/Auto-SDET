"""Tests for conversions/hexadecimal_to_decimal.py."""

import pytest

from conversions.hexadecimal_to_decimal import hex_table, hex_to_decimal


class TestHexTable:
    """Tests for the module-level hex_table mapping."""

    def test_hex_table_has_sixteen_entries(self):
        """The lookup table must contain exactly 16 entries (0-F)."""
        assert len(hex_table) == 16

    def test_hex_table_maps_digits_and_letters(self):
        """Digits and lowercase letters must map to their decimal values."""
        assert hex_table["0"] == 0
        assert hex_table["9"] == 9
        assert hex_table["a"] == 10
        assert hex_table["f"] == 15

    def test_hex_table_has_no_uppercase_keys(self):
        """The table keys are lowercase only; uppercase is handled by lowering."""
        assert "A" not in hex_table
        assert "F" not in hex_table


class TestHexToDecimalHappyPath:
    """Positive test cases for hex_to_decimal."""

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("0", 0),
            ("1", 1),
            ("9", 9),
            ("a", 10),
            ("f", 15),
            ("10", 16),
            ("12f", 303),
            ("ff", 255),
            ("100", 256),
            ("deadbeef", 3735928559),
        ],
    )
    def test_single_and_multi_digit_values(self, hex_string, expected):
        """Lowercase hex strings convert to the expected decimal integers."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("A", 10),
            ("F", 15),
            ("FfFf", 65535),
            ("DEADBEEF", 3735928559),
            ("AbC", 2748),
        ],
    )
    def test_uppercase_and_mixed_case(self, hex_string, expected):
        """Uppercase and mixed-case input is accepted and converted correctly."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("   12f   ", 303),
            ("\tff\n", 255),
            ("  A  ", 10),
            ("\n\n0\n\n", 0),
        ],
    )
    def test_surrounding_whitespace_is_stripped(self, hex_string, expected):
        """Leading and trailing whitespace is ignored."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("-Ff", -255),
            ("-a", -10),
            ("-0", 0),
            ("-1", -1),
            ("  -12f  ", -303),
            ("-DEADBEEF", -3735928559),
        ],
    )
    def test_negative_values(self, hex_string, expected):
        """A leading minus sign produces a negative decimal result."""
        assert hex_to_decimal(hex_string) == expected

    def test_negative_zero_is_zero(self):
        """'-0' must not produce a negative zero artifact (int has no -0)."""
        result = hex_to_decimal("-0")
        assert result == 0
        assert isinstance(result, int)

    def test_large_value(self):
        """Very large hex strings convert without overflow."""
        assert hex_to_decimal("f" * 20) == 16**20 - 1

    def test_return_type_is_int(self):
        """The function always returns a plain int."""
        assert isinstance(hex_to_decimal("ff"), int)


class TestHexToDecimalErrors:
    """Edge cases and error handling for hex_to_decimal."""

    @pytest.mark.parametrize("hex_string", ["", "   ", "\t", "\n", "  \n\t "])
    def test_empty_or_whitespace_only_raises(self, hex_string):
        """Empty or whitespace-only input raises ValueError with a clear message."""
        with pytest.raises(ValueError) as exc_info:
            hex_to_decimal(hex_string)
        assert "Empty string was passed to the function" in str(exc_info.value)

    @pytest.mark.parametrize(
        "hex_string",
        [
            "12m",
            "g",
            "F-f",
            "0x1f",
            "1.5",
            "12 34",
            "abc!",
            "--1",
            "+1",
            "1_0",
        ],
    )
    def test_non_hexadecimal_input_raises(self, hex_string):
        """Invalid characters raise ValueError with the non-hex message."""
        with pytest.raises(ValueError) as exc_info:
            hex_to_decimal(hex_string)
        assert "Non-hexadecimal value was passed to the function" in str(exc_info.value)

    def test_lone_minus_sign_returns_zero(self):
        """A bare '-' leaves an empty body, which sums to zero."""
        assert hex_to_decimal("-") == 0

    def test_non_string_input_raises_attribute_error(self):
        """Passing a non-string (no .strip) surfaces an AttributeError."""
        with pytest.raises(AttributeError):
            hex_to_decimal(123)

    def test_none_input_raises_attribute_error(self):
        """Passing None surfaces an AttributeError from .strip()."""
        with pytest.raises(AttributeError):
            hex_to_decimal(None)


class TestHexToDecimalRoundTrip:
    """Cross-check against Python's built-in int(..., 16)."""

    @pytest.mark.parametrize(
        "value",
        [0, 1, 15, 16, 255, 4096, 123456789, 2**64 - 1],
    )
    def test_matches_builtin_int_conversion(self, value):
        """hex_to_decimal agrees with int(hex(value), 16) for many values."""
        assert hex_to_decimal(hex(value)[2:]) == value

    @pytest.mark.parametrize(
        "value",
        [-1, -15, -255, -123456789],
    )
    def test_matches_builtin_for_negatives(self, value):
        """Negative values round-trip through hex formatting and back."""
        hex_string = hex(value).replace("0x", "")
        assert hex_to_decimal(hex_string) == value
