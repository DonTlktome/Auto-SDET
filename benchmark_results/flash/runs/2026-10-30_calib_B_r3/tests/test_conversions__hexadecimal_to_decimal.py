"""Tests for conversions/hexadecimal_to_decimal.py."""

import pytest

from conversions.hexadecimal_to_decimal import hex_table, hex_to_decimal


class TestHexTable:
    """Tests for the module-level hex_table lookup dictionary."""

    def test_hex_table_has_sixteen_entries(self):
        """The lookup table must contain exactly 16 entries (0-9, a-f)."""
        assert len(hex_table) == 16

    def test_hex_table_maps_digits_correctly(self):
        """Each key must map to its integer value."""
        assert hex_table["0"] == 0
        assert hex_table["9"] == 9
        assert hex_table["a"] == 10
        assert hex_table["f"] == 15

    def test_hex_table_keys_are_lowercase(self):
        """Keys must be lowercase single characters without the '0x' prefix."""
        for key in hex_table:
            assert len(key) == 1
            assert key == key.lower()


class TestHexToDecimalPositive:
    """Happy-path tests for hex_to_decimal."""

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
            ("0000", 0),
            ("00012f", 303),
        ],
    )
    def test_valid_hex_strings(self, hex_string, expected):
        """Valid hexadecimal strings convert to the expected decimal value."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("   12f   ", 303),
            ("\t12f\n", 303),
            ("  a  ", 10),
            ("\n\nFfFf\t", 65535),
        ],
    )
    def test_surrounding_whitespace_is_stripped(self, hex_string, expected):
        """Leading and trailing whitespace must be ignored."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("A", 10),
            ("a", 10),
            ("Ff", 255),
            ("fF", 255),
            ("AbCdEf", 11259375),
        ],
    )
    def test_case_insensitivity(self, hex_string, expected):
        """Upper and mixed case input must produce the same result."""
        assert hex_to_decimal(hex_string) == expected


class TestHexToDecimalNegative:
    """Tests for negative hexadecimal input."""

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("-0", 0),
            ("-1", -1),
            ("-a", -10),
            ("-Ff", -255),
            ("-12f", -303),
            ("-FfFf", -65535),
        ],
    )
    def test_negative_hex_strings(self, hex_string, expected):
        """A leading minus sign yields a negative decimal value."""
        assert hex_to_decimal(hex_string) == expected

    @pytest.mark.parametrize(
        "hex_string, expected",
        [
            ("  -Ff  ", -255),
            ("\t-a\n", -10),
        ],
    )
    def test_negative_with_whitespace(self, hex_string, expected):
        """Whitespace around a negative hex string is stripped before parsing."""
        assert hex_to_decimal(hex_string) == expected

    def test_negative_zero_is_zero(self):
        """'-0' must not produce a negative zero (int has no -0)."""
        result = hex_to_decimal("-0")
        assert result == 0
        assert result == -result


class TestHexToDecimalErrors:
    """Error-handling tests for hex_to_decimal."""

    @pytest.mark.parametrize("hex_string", ["", "   ", "\t", "\n", "  \t\n  "])
    def test_empty_or_whitespace_only_raises_value_error(self, hex_string):
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
            "abc!",
            "--1",
            "+1",
            "1_0",
        ],
    )
    def test_non_hexadecimal_raises_value_error(self, hex_string):
        """Non-hexadecimal characters raise ValueError with the non-hex message."""
        with pytest.raises(ValueError) as exc_info:
            hex_to_decimal(hex_string)
        assert str(exc_info.value) == "Non-hexadecimal value was passed to the function"

    def test_negative_sign_only_returns_zero(self):
        """A lone '-' is stripped to an empty body, which yields 0 (negated zero)."""
        result = hex_to_decimal("-")
        assert result == 0
        assert result == -result

    def test_non_string_input_raises_attribute_error(self):
        """Passing a non-string (no .strip) raises AttributeError."""
        with pytest.raises(AttributeError):
            hex_to_decimal(123)


class TestHexToDecimalBoundaries:
    """Boundary and large-value tests."""

    def test_single_max_digit(self):
        """The largest single hex digit 'f' equals 15."""
        assert hex_to_decimal("f") == 15

    def test_large_value(self):
        """A long hex string converts correctly (16**8 - 1)."""
        assert hex_to_decimal("ffffffff") == 16**8 - 1

    def test_large_negative_value(self):
        """A long negative hex string converts correctly."""
        assert hex_to_decimal("-ffffffff") == -(16**8 - 1)

    @staticmethod
    def _from_builtin_hex(value: int) -> str:
        """Reproduce builtin hex() output without the unsupported '0x' prefix."""
        text = hex(value)
        sign = ""
        if text.startswith("-"):
            sign = "-"
            text = text[1:]
        assert text.startswith("0x")
        return sign + text[2:]

    def test_roundtrip_with_builtin_hex(self):
        """Values produced by builtin hex() round-trip back to the original int."""
        for value in (0, 1, 15, 16, 255, 4096, 123456789):
            assert hex_to_decimal(self._from_builtin_hex(value)) == value
            assert hex_to_decimal(self._from_builtin_hex(-value)) == -value
