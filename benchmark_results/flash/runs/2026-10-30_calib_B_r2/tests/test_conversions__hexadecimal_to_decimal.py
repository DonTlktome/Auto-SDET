"""Tests for conversions/hexadecimal_to_decimal.py."""

import pytest

from conversions.hexadecimal_to_decimal import hex_table, hex_to_decimal


class TestHexTable:
    """Tests for the module-level hex_table lookup dictionary."""

    def test_hex_table_has_sixteen_entries(self):
        """The lookup table must contain exactly 16 entries (0-f)."""
        assert len(hex_table) == 16

    def test_hex_table_maps_digits_correctly(self):
        """Each key must map to its integer value."""
        expected = {
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
            "8": 8, "9": 9, "a": 10, "b": 11, "c": 12, "d": 13, "e": 14, "f": 15,
        }
        assert hex_table == expected

    def test_hex_table_keys_are_lowercase(self):
        """Keys must be lowercase single characters without the '0x' prefix."""
        for key in hex_table:
            assert len(key) == 1
            assert key == key.lower()


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
            ("FfFf", 65535),
            ("deadbeef", 3735928559),
            ("0" * 10, 0),
        ],
    )
    def test_valid_hex_strings(self, hex_string, expected):
        """Valid hexadecimal strings convert to the expected decimal value."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("-1", -1),
            ("-a", -10),
            ("-Ff", -255),
            ("-12f", -303),
            ("-0", 0),
        ],
    )
    def test_negative_hex_strings(self, hex_string, expected):
        """Negative hexadecimal strings convert to negative decimals."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("   12f   ", 303),
            ("\t12f\n", 303),
            ("  -Ff  ", -255),
            ("\n\nA\t", 10),
        ],
    )
    def test_whitespace_is_stripped(self, hex_string, expected):
        """Leading and trailing whitespace is ignored."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("A", 10),
            ("FF", 255),
            ("aBcD", 43981),
            ("DeAdBeEf", 3735928559),
        ],
    )
    def test_case_insensitivity(self, hex_string, expected):
        """Mixed and upper case input is handled identically to lowercase."""
        assert hex_to_decimal(hex_string) == expected

    def test_matches_builtin_int_conversion(self):
        """Result must agree with Python's built-in int(x, 16)."""
        for value in (0, 1, 15, 16, 255, 4096, 123456789):
            assert hex_to_decimal(hex(value)[2:]) == value


class TestHexToDecimalEdgeCases:
    """Boundary and error cases for hex_to_decimal."""

    @pytest.mark.parametrize("hex_string", ["", "   ", "\t\n", " \r\n "])
    def test_empty_or_whitespace_only_raises(self, hex_string):
        """Empty or whitespace-only input raises ValueError with the empty message."""
        with pytest.raises(ValueError) as exc_info:
            hex_to_decimal(hex_string)
        assert str(exc_info.value) == "Empty string was passed to the function"

    @pytest.mark.parametrize(
        "hex_string",
        [
            "12m",
            "g",
            "F-f",
            "0x1f",
            "1.5",
            "12 3",
            "--1",
            "1-",
            "abc!",
            "\u0661\u0662",  # non-ASCII digits
        ],
    )
    def test_non_hexadecimal_raises(self, hex_string):
        """Non-hexadecimal input raises ValueError with the non-hex message."""
        with pytest.raises(ValueError) as exc_info:
            hex_to_decimal(hex_string)
        assert str(exc_info.value) == "Non-hexadecimal value was passed to the function"

    def test_single_minus_sign_returns_zero(self):
        """A lone '-' leaves an empty body after the sign is stripped, which sums to zero."""
        assert hex_to_decimal("-") == 0
        assert isinstance(hex_to_decimal("-"), int)
        assert not isinstance(hex_to_decimal("-"), bool)

    def test_large_value(self):
        """Very large hexadecimal values are converted without overflow."""
        assert hex_to_decimal("f" * 20) == int("f" * 20, 16)

    def test_leading_zeros_preserved_value(self):
        """Leading zeros do not change the numeric value."""
        assert hex_to_decimal("0000ff") == 255
        assert hex_to_decimal("-0000ff") == -255

    def test_input_string_not_mutated_by_caller(self):
        """The function must not modify the caller's string (strings are immutable)."""
        original = "  12f  "
        hex_to_decimal(original)
        assert original == "  12f  "

    def test_return_type_is_int(self):
        """The return value is always a Python int."""
        result = hex_to_decimal("ff")
        assert isinstance(result, int)
        assert not isinstance(result, bool)
