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


def test_decimal_to_binary_iterative_large_value():
    """A large integer produces the expected binary representation."""
    assert decimal_to_binary_iterative(2**20) == "0b1" + "0" * 20


@pytest.mark.parametrize("bad_value", [16.16, 0.0, -3.5])
def test_decimal_to_binary_iterative_float_raises(bad_value):
    """Floats raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        decimal_to_binary_iterative(bad_value)
    assert "'float' object cannot be interpreted as an integer" in str(exc_info.value)


@pytest.mark.parametrize("bad_value", ["0xfffff", "", "42"])
def test_decimal_to_binary_iterative_str_raises(bad_value):
    """Strings raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        decimal_to_binary_iterative(bad_value)
    assert "'str' object cannot be interpreted as an integer" in str(exc_info.value)


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
        (255, "11111111"),
    ],
)
def test_decimal_to_binary_recursive_helper_ints(decimal, expected):
    """Non-negative integers convert to a bare binary string (no prefix)."""
    assert decimal_to_binary_recursive_helper(decimal) == expected


@pytest.mark.parametrize(
    "decimal, expected",
    [
        ("72", "1001000"),
        ("0", "0"),
        ("1", "1"),
        ("1000", "1111101000"),
    ],
)
def test_decimal_to_binary_recursive_helper_numeric_strings(decimal, expected):
    """Numeric strings are accepted because the helper calls int() first."""
    assert decimal_to_binary_recursive_helper(decimal) == expected


def test_decimal_to_binary_recursive_helper_invalid_string_raises():
    """A non-numeric string raises ValueError from int()."""
    with pytest.raises(ValueError) as exc_info:
        decimal_to_binary_recursive_helper("number")
    assert "invalid literal for int()" in str(exc_info.value)


def test_decimal_to_binary_recursive_helper_float_truncates():
    """Floats are truncated by int() rather than rejected."""
    assert decimal_to_binary_recursive_helper(2.9) == "10"


# ---------------------------------------------------------------------------
# decimal_to_binary_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (0, "0b0"),
        (1, "0b1"),
        (2, "0b10"),
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
        ("  -40  ", "-0b101000"),
    ],
)
def test_decimal_to_binary_recursive_negative(number, expected):
    """Negative integers (and numeric strings) get a '-0b' prefix."""
    assert decimal_to_binary_recursive(number) == expected


@pytest.mark.parametrize("bad_value", [40.8, "forty", "4.5", "0x1f", "1 2"])
def test_decimal_to_binary_recursive_invalid_raises(bad_value):
    """Non-integer inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError) as exc_info:
        decimal_to_binary_recursive(bad_value)
    assert "Input value is not an integer" in str(exc_info.value)


@pytest.mark.parametrize("empty_value", ["", "   ", "\t\n"])
def test_decimal_to_binary_recursive_empty_raises(empty_value):
    """Blank or whitespace-only input raises the 'No input value' ValueError."""
    with pytest.raises(ValueError) as exc_info:
        decimal_to_binary_recursive(empty_value)
    assert "No input value was provided" in str(exc_info.value)


def test_decimal_to_binary_recursive_only_minus_sign_raises():
    """A lone '-' strips to an empty string and is rejected as non-numeric."""
    with pytest.raises(ValueError) as exc_info:
        decimal_to_binary_recursive("-")
    assert "Input value is not an integer" in str(exc_info.value)


def test_decimal_to_binary_recursive_large_value():
    """A large power of two round-trips correctly."""
    assert decimal_to_binary_recursive(2**16) == "0b1" + "0" * 16


def test_iterative_and_recursive_agree():
    """Both implementations agree on the same inputs."""
    for value in [0, 1, 5, 17, 128, 999, -1, -64, -1023]:
        assert decimal_to_binary_iterative(value) == decimal_to_binary_recursive(value)