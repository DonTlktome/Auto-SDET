"""Tests for conversions/decimal_to_binary.py."""

import sys

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
        (3, "0b11"),
        (7, "0b111"),
        (8, "0b1000"),
        (35, "0b100011"),
        (1000, "0b1111101000"),
        (255, "0b11111111"),
        (256, "0b100000000"),
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
        (-40, "-0b101000"),
        (-1000, "-0b1111101000"),
    ],
)
def test_decimal_to_binary_iterative_negative(num, expected):
    """Negative integers are converted with a '-0b' prefix."""
    assert decimal_to_binary_iterative(num) == expected


def test_decimal_to_binary_iterative_zero_is_special_case():
    """Zero returns the literal '0b0' rather than an empty digit string."""
    assert decimal_to_binary_iterative(0) == "0b0"


def test_decimal_to_binary_iterative_large_number():
    """A large integer is converted correctly (boundary / stress case)."""
    assert decimal_to_binary_iterative(2**64) == "0b1" + "0" * 64


@pytest.mark.parametrize("bad_input", [16.16, 0.0, -3.5, 1e10])
def test_decimal_to_binary_iterative_rejects_floats(bad_input):
    """Float inputs raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        decimal_to_binary_iterative(bad_input)
    assert "'float' object cannot be interpreted as an integer" in str(exc_info.value)


@pytest.mark.parametrize("bad_input", ["0xfffff", "", "42", "abc"])
def test_decimal_to_binary_iterative_rejects_strings(bad_input):
    """String inputs raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        decimal_to_binary_iterative(bad_input)
    assert "'str' object cannot be interpreted as an integer" in str(exc_info.value)


def test_decimal_to_binary_iterative_rejects_none():
    """None is not a valid integer and raises TypeError."""
    with pytest.raises(TypeError):
        decimal_to_binary_iterative(None)


def test_decimal_to_binary_iterative_returns_str():
    """The return type is always str."""
    assert isinstance(decimal_to_binary_iterative(42), str)


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
        (4, "100"),
        (7, "111"),
        (8, "1000"),
        (1000, "1111101000"),
        (1024, "10000000000"),
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
        ("255", "11111111"),
    ],
)
def test_decimal_to_binary_recursive_helper_numeric_strings(decimal, expected):
    """Numeric strings are accepted because the helper calls int() first."""
    assert decimal_to_binary_recursive_helper(decimal) == expected


def test_decimal_to_binary_recursive_helper_zero_and_one_base_cases():
    """The recursion exit cases for 0 and 1 return their own string form."""
    assert decimal_to_binary_recursive_helper(0) == "0"
    assert decimal_to_binary_recursive_helper(1) == "1"


@pytest.mark.parametrize("bad_input", ["number", "", "1.5", "abc123"])
def test_decimal_to_binary_recursive_helper_invalid_strings(bad_input):
    """Non-numeric strings raise ValueError from int()."""
    with pytest.raises(ValueError):
        decimal_to_binary_recursive_helper(bad_input)


def test_decimal_to_binary_recursive_helper_float_truncates():
    """A float is truncated by int() rather than rejected (documented behaviour)."""
    assert decimal_to_binary_recursive_helper(4.9) == "100"


def test_decimal_to_binary_recursive_helper_negative_recurses_forever():
    """Negative input never reaches a base case, so recursion overflows."""
    with pytest.raises(RecursionError):
        decimal_to_binary_recursive_helper(-5)


# ---------------------------------------------------------------------------
# decimal_to_binary_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (0, "0b0"),
        (1, "0b1"),
        (2, "0b10"),
        (7, "0b111"),
        (40, "0b101000"),
        (1000, "0b1111101000"),
        ("40", "0b101000"),
        ("  40  ", "0b101000"),
        ("0", "0b0"),
    ],
)
def test_decimal_to_binary_recursive_positive(number, expected):
    """Positive integers (and numeric strings) get a '0b' prefix."""
    assert decimal_to_binary_recursive(number) == expected


@pytest.mark.parametrize(
    "number, expected",
    [
        (-1, "-0b1"),
        (-2, "-0b10"),
        (-40, "-0b101000"),
        ("-40", "-0b101000"),
        (" -40 ", "-0b101000"),
        (-1000, "-0b1111101000"),
    ],
)
def test_decimal_to_binary_recursive_negative(number, expected):
    """Negative integers (and numeric strings) get a '-0b' prefix."""
    assert decimal_to_binary_recursive(number) == expected


def test_decimal_to_binary_recursive_zero():
    """Zero is handled as a positive value and yields '0b0'."""
    assert decimal_to_binary_recursive(0) == "0b0"


@pytest.mark.parametrize("bad_input", [40.8, -0.5, 1.0, "forty", "4.5", "0x1f", "1 2"])
def test_decimal_to_binary_recursive_rejects_non_integers(bad_input):
    """Floats and non-numeric strings raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        decimal_to_binary_recursive(bad_input)
    assert "Input value is not an integer" in str(exc_info.value)


@pytest.mark.parametrize("empty_input", ["", "   ", "\t", "\n"])
def test_decimal_to_binary_recursive_rejects_empty_input(empty_input):
    """Blank or whitespace-only input raises the dedicated empty-input error."""
    with pytest.raises(ValueError) as exc_info:
        decimal_to_binary_recursive(empty_input)
    assert "No input value was provided" in str(exc_info.value)


def test_decimal_to_binary_recursive_rejects_none():
    """None stringifies to 'None', which is not numeric, so ValueError is raised."""
    with pytest.raises(ValueError) as exc_info:
        decimal_to_binary_recursive(None)
    assert "Input value is not an integer" in str(exc_info.value)


def test_decimal_to_binary_recursive_returns_str():
    """The return type is always str."""
    assert isinstance(decimal_to_binary_recursive(42), str)


def test_decimal_to_binary_recursive_uses_helper(monkeypatch):
    """The public function delegates digit generation to the recursive helper."""
    calls = []

    def fake_helper(decimal):
        calls.append(decimal)
        return "101010"

    # Resolve the module object that actually defines the functions instead of
    # relying on dotted-name resolution, which requires importing the package.
    module = sys.modules[decimal_to_binary_recursive.__module__]
    monkeypatch.setattr(module, "decimal_to_binary_recursive_helper", fake_helper)
    assert decimal_to_binary_recursive(-42) == "-0b101010"
    assert calls == [42]


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num", [0, 1, 2, 5, 13, 64, 255, 1024, 123456])
def test_iterative_and_recursive_agree(num):
    """Both implementations produce identical output for the same input."""
    assert decimal_to_binary_iterative(num) == decimal_to_binary_recursive(num)
    assert decimal_to_binary_iterative(-num) == decimal_to_binary_recursive(-num)
