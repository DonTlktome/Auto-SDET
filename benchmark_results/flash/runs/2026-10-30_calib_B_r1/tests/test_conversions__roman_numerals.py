"""Tests for conversions/roman_numerals.py."""

import pytest

from conversions.roman_numerals import ROMAN, int_to_roman, roman_to_int


# ---------------------------------------------------------------------------
# ROMAN table
# ---------------------------------------------------------------------------


def test_roman_table_is_descending_and_complete():
    """The ROMAN lookup table must be sorted descending and cover all values."""
    values = [value for value, _ in ROMAN]
    assert values == sorted(values, reverse=True)
    assert values[0] == 1000
    assert values[-1] == 1
    assert len(values) == 13


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
    """Known roman numerals convert to their documented integer values."""
    assert roman_to_int(roman) == expected


@pytest.mark.parametrize(
    "roman, expected",
    [
        ("IIII", 4),  # non-canonical but additive
        ("VV", 10),  # repeated V
        ("XXXX", 40),
        ("MMMM", 4000),  # above documented range, still additive
    ],
)
def test_roman_to_int_non_canonical_inputs(roman, expected):
    """Non-canonical (but parseable) numerals are summed additively."""
    assert roman_to_int(roman) == expected


def test_roman_to_int_empty_string_returns_zero():
    """An empty string has no symbols, so the total is 0."""
    assert roman_to_int("") == 0


def test_roman_to_int_single_character_boundaries():
    """The smallest and largest single symbols map to 1 and 1000."""
    assert roman_to_int("I") == 1
    assert roman_to_int("M") == 1000


def test_roman_to_int_invalid_character_raises_key_error():
    """An unknown symbol is not in the lookup dict and raises KeyError."""
    with pytest.raises(KeyError):
        roman_to_int("ABC")


def test_roman_to_int_lowercase_raises_key_error():
    """Lowercase symbols are not supported and raise KeyError."""
    with pytest.raises(KeyError):
        roman_to_int("xiv")


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
    """Known integers convert to their documented roman numerals."""
    assert int_to_roman(number) == expected


def test_int_to_roman_zero_returns_empty_string():
    """Zero has no roman representation; the function returns an empty string."""
    assert int_to_roman(0) == ""


def test_int_to_roman_negative_input_is_not_supported():
    """Negative integers are outside the documented domain of the function.

    ``int_to_roman`` does no range checking, so a negative dividend makes
    ``divmod`` return a negative factor followed by the complement
    (``divmod(-5, 1000) == (-1, 995)``). The empty ``"M" * -1`` block is
    dropped and the complement is encoded, so ``-5`` renders as ``"CMXCV"``
    (i.e. 995) rather than a meaningful numeral.
    """
    assert int_to_roman(-5) == "CMXCV"


def test_int_to_roman_above_range_is_additive():
    """Values above 3999 are still rendered by repeating M."""
    assert int_to_roman(4000) == "MMMM"


def test_int_to_roman_does_not_mutate_input():
    """The integer argument is immutable; the caller's value is unchanged."""
    number = 1994
    result = int_to_roman(number)
    assert result == "MCMXCIV"
    assert number == 1994


# ---------------------------------------------------------------------------
# Round-trip properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [1, 2, 7, 14, 49, 99, 444, 999, 1234, 2024, 3999])
def test_round_trip_int_to_roman_to_int(number):
    """Converting to roman and back returns the original integer."""
    assert roman_to_int(int_to_roman(number)) == number


@pytest.mark.parametrize(
    "roman", ["I", "IV", "IX", "XL", "XC", "CD", "CM", "MMXXIV", "MMMCMXCIX"]
)
def test_round_trip_roman_to_int_to_roman(roman):
    """Canonical numerals survive a round trip through int_to_roman."""
    assert int_to_roman(roman_to_int(roman)) == roman


def test_round_trip_all_values_in_range():
    """Every integer from 1 to 3999 round-trips exactly."""
    for number in range(1, 4000):
        assert roman_to_int(int_to_roman(number)) == number
