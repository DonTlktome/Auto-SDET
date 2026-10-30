# AUTO-SDET: Test generation FAILED after 3 retries
# Last error: 

"""Tests for conversions/decimal_to_binary.py."""

import pytest

from conversions.decimal_to_binary import (
    decimal_to_binary_iterative,
    decimal_to_binary_recursive,
    decimal_to_binary_recursive_helper,
)


# ---------------------------------------------------------------------------
# decimal_to_binary_iterative
# ---------------------------------------------------------------------------


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
def test_decimal_to_binary_iterative_positive(num, expected):
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
def test_decimal_to_binary_iterative_negative(num, expected):
    """Negative integers are converted with a '-0b' prefix."""
    assert decimal_to_binary_iterative(num) == expected


def test_decimal_to_binary_iterative_zero_edge_case():
    """Zero is a boundary case handled by an early return."""
    assert decimal_to_binary_iterative(0) == "0b0"


def test_decimal_to_binary_iterative_matches_builtin_bin():
    """The result must agree with Python's built-in bin() for many values."""
    for value in list(range(-50, 51)) + [12345, -98765]:
        assert decimal_to_binary_iterative(value) == bin(value)


@pytest.mark.parametrize("bad_value", [16.16, 0.0, -3.5, float("inf")])
def test_decimal_to_binary_iterative_float_raises(bad_value):
    """Floats raise TypeError with the documented message."""
    with pytest.raises(
        TypeError, match="'float' object cannot be interpreted as an integer"
    ):
        decimal_to_binary_iterative(bad_value)


@pytest.mark.parametrize("bad_value", ["0xfffff", "", "42", "abc"])
def test_decimal_to_binary_iterative_str_raises(bad_value):
    """Strings raise TypeError with the documented message."""
    with pytest.raises(
        TypeError, match="'str' object cannot be interpreted as an integer"
    ):
        decimal_to_binary_iterative(bad_value)


def test_decimal_to_binary_iterative_bool_input():
    """Booleans are ints in Python, so True -> '0b1' and False -> '0b0'."""
    assert decimal_to_binary_iterative(True) == "0b1"
    assert decimal_to_binary_iterative(False) == "0b0"


# ---------------------------------------------------------------------------
# decimal_to_binary_recursive_helper
# ---------------------------------------------------------------------------


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
def test_decimal_to_binary_recursive_helper(decimal, expected):
    """The helper returns a bare binary string without any prefix."""
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
def test_decimal_to_binary_recursive_helper_numeric_strings(decimal, expected):
    """Numeric strings are accepted because the helper calls int() first."""
    assert decimal_to_binary_recursive_helper(decimal) == expected


def test_decimal_to_binary_recursive_helper_invalid_string_raises():
    """A non-numeric string propagates the ValueError from int()."""
    with pytest.raises(ValueError, match="invalid literal for int"):
        decimal_to_binary_recursive_helper("number")


def test_decimal_to_binary_recursive_helper_float_truncates():
    """Floats are truncated by int(), so 7.9 becomes '111'."""
    assert decimal_to_binary_recursive_helper(7.9) == "111"


# ---------------------------------------------------------------------------
# decimal_to_binary_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (0, "0b0"),
        (1, "0b1"),
        (40, "0b101000"),
        (1000, "0b1111101000"),
        ("40", "0b101000"),
        ("  40  ", "0b101000"),
    ],
)
def test_decimal_to_binary_recursive_positive(number, expected):
    """Positive integers (and numeric strings) get a '0b' prefix."""
    assert decimal_to_binary_recursive(number) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (-1, "-0b1"),
        (-40, "-0b101000"),
        ("-40", "-0b101000"),
        ("-0", "0b0"),
    ],
)
def test_decimal_to_binary_recursive_negative(number, expected):
    """Negative integers get a '-0b' prefix; '-0' normalises to '0b0'."""
    assert decimal_to_binary_recursive(number) == expected


def test_decimal_to_binary_recursive_matches_builtin_bin():
    """The recursive variant agrees with bin() across a range of values."""
    for value in range(-64, 65):
        assert decimal_to_binary_recursive(value) == bin(value)


@pytest.mark.parametrize("bad_value", [40.8, "forty", "4.5", "1e3", "0x1f"])
def test_decimal_to_binary_recursive_invalid_raises(bad_value):
    """Non-integer inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="Input value is not an integer"):
        decimal_to_binary_recursive(bad_value)


@pytest.mark.parametrize("empty_value", ["", "   ", "\t\n"])
def test_decimal_to_binary_recursive_empty_raises(empty_value):
    """Blank or whitespace-only input raises the 'No input value' error."""
    with pytest.raises(ValueError, match="No input value was provided"):
        decimal_to_binary_recursive(empty_value)


def test_decimal_to_binary_recursive_none_raises():
    """None stringifies to 'None', which is not numeric -> ValueError."""
    with pytest.raises(ValueError, match="Input value is not an integer"):
        decimal_to_binary_recursive(None)


def test_decimal_to_binary_recursive_large_value():
    """A large integer is converted correctly (recursion depth is fine)."""
    assert decimal_to_binary_recursive(2**20) == bin(2**20)