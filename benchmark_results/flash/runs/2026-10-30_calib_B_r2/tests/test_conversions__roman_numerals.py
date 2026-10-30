"""Tests for conversions/roman_numerals.py."""

import pytest

from conversions.roman_numerals import ROMAN, int_to_roman, roman_to_int


# ---------------------------------------------------------------------------
# ROMAN table
# ---------------------------------------------------------------------------


def test_roman_table_is_sorted_descending():
    """The ROMAN lookup table must be ordered from largest to smallest value."""
    values = [value for value, _ in ROMAN]
    assert values == sorted(values, reverse=True)


def test_roman_table_boundaries():
    """The ROMAN table must start at 1000/M and end at 1/I."""
    assert ROMAN[0] == (1000, "M")
    assert ROMAN[-1] == (1, "I")


# ---------------------------------------------------------------------------
# roman_to_int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "roman, expected",
    [
        ("I", 1),
        ("III", 3),
        ("IV", 4),
        ("V", 5),
        ("IX", 9),
        ("X", 10),
        ("XL", 40),
        ("L", 50),
        ("XC", 90),
        ("C", 100),
        ("CD", 400),
        ("D", 500),
        ("CM", 900),
        ("M", 1000),
        ("CLIV", 154),
        ("MIX", 1009),
        ("MMD", 2500),
        ("MMMCMXCIX", 3999),
    ],
)
def test_roman_to_int_happy_path(roman, expected):
    """Each canonical roman numeral converts to its expected integer value."""
    assert roman_to_int(roman) == expected


def test_roman_to_int_single_character_minimum():
    """The smallest valid input 'I' maps to 1."""
    assert roman_to_int("I") == 1


def test_roman_to_int_maximum_value():
    """The largest valid input 'MMMCMXCIX' maps to 3999."""
    assert roman_to_int("MMMCMXCIX") == 3999


def test_roman_to_int_empty_string_returns_zero():
    """An empty string has no symbols, so the total is 0."""
    assert roman_to_int("") == 0


def test_roman_to_int_repeated_symbols():
    """Repeated additive symbols accumulate (e.g. 'XXX' -> 30)."""
    assert roman_to_int("XXX") == 30


def test_roman_to_int_subtractive_pair_at_end():
    """A subtractive pair at the very end of the string is handled correctly."""
    assert roman_to_int("XIV") == 14


def test_roman_to_int_invalid_symbol_raises_key_error():
    """An unknown symbol is not in the value map and raises KeyError."""
    with pytest.raises(KeyError):
        roman_to_int("A")


def test_roman_to_int_lowercase_raises_key_error():
    """Lowercase symbols are not supported and raise KeyError."""
    with pytest.raises(KeyError):
        roman_to_int("iii")


# ---------------------------------------------------------------------------
# int_to_roman
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, "I"),
        (3, "III"),
        (4, "IV"),
        (5, "V"),
        (9, "IX"),
        (10, "X"),
        (40, "XL"),
        (50, "L"),
        (90, "XC"),
        (100, "C"),
        (400, "CD"),
        (500, "D"),
        (900, "CM"),
        (1000, "M"),
        (154, "CLIV"),
        (1009, "MIX"),
        (2500, "MMD"),
        (3999, "MMMCMXCIX"),
    ],
)
def test_int_to_roman_happy_path(number, expected):
    """Each integer converts to its expected roman numeral."""
    assert int_to_roman(number) == expected


def test_int_to_roman_minimum_value():
    """The smallest supported value 1 converts to 'I'."""
    assert int_to_roman(1) == "I"


def test_int_to_roman_maximum_value():
    """The largest supported value 3999 converts to 'MMMCMXCIX'."""
    assert int_to_roman(3999) == "MMMCMXCIX"


def test_int_to_roman_zero_returns_empty_string():
    """Zero has no roman representation, so an empty string is returned."""
    assert int_to_roman(0) == ""


def test_int_to_roman_negative_is_not_validated():
    """Negative inputs are not validated by int_to_roman.

    divmod keeps the sign on the quotient, so the negative factor produces no
    symbols for that place and the remaining (positive) remainder is encoded
    normally: -5 -> divmod(-5, 1000) == (-1, 995) -> 'CMXCV'.
    """
    assert int_to_roman(-5) == "CMXCV"


def test_int_to_roman_above_range_uses_repeated_m():
    """Values above 3999 are not validated and simply repeat 'M'."""
    assert int_to_roman(4000) == "MMMM"


# ---------------------------------------------------------------------------
# Round-trip properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [1, 2, 7, 14, 49, 99, 444, 1666, 2024, 3999])
def test_round_trip_int_to_roman_to_int(number):
    """Converting to roman and back must return the original integer."""
    assert roman_to_int(int_to_roman(number)) == number


@pytest.mark.parametrize(
    "roman", ["I", "IV", "IX", "XLII", "CDXLIV", "MMXXIV", "MMMCMXCIX"]
)
def test_round_trip_roman_to_int_to_roman(roman):
    """Converting to int and back must return the original roman numeral."""
    assert int_to_roman(roman_to_int(roman)) == roman


def test_round_trip_all_values_in_range():
    """Every integer in 1..3999 survives a full round trip."""
    for number in range(1, 4000):
        assert roman_to_int(int_to_roman(number)) == number
