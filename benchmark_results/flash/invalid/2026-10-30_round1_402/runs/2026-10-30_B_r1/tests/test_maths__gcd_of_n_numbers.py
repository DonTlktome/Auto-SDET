"""Tests for maths/gcd_of_n_numbers.py"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcd_of_n_numbers import get_factors, get_greatest_common_divisor


# ---------------------------------------------------------------------------
# get_factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (45, Counter({3: 2, 5: 1})),
        (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
        (23, Counter({23: 1})),
        (2, Counter({2: 1})),
        (4, Counter({2: 2})),
        (100, Counter({2: 2, 5: 2})),
        (97, Counter({97: 1})),
    ],
)
def test_get_factors_happy_path(number, expected):
    """get_factors returns the correct prime factorisation for positive ints."""
    assert get_factors(number) == expected


def test_get_factors_one_returns_counter_with_one():
    """The special case number == 1 returns Counter({1: 1})."""
    assert get_factors(1) == Counter({1: 1})


def test_get_factors_accepts_prepopulated_counter():
    """A pre-populated Counter is mutated/extended rather than discarded."""
    seed = Counter({2: 1})
    result = get_factors(45, factors=seed)
    assert result[3] == 2
    assert result[5] == 1
    assert result[2] == 1


def test_get_factors_accepts_custom_starting_factor():
    """Starting the search at a higher factor skips smaller primes."""
    result = get_factors(45, factor=3)
    assert result == Counter({3: 2, 5: 1})


@pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, "10", None])
def test_get_factors_invalid_input_raises_type_error(bad_value):
    """Non-positive or non-integer inputs raise TypeError."""
    with pytest.raises(TypeError):
        get_factors(bad_value)


def test_get_factors_error_message():
    """The TypeError carries the documented message."""
    with pytest.raises(TypeError, match="number must be integer and greater than zero"):
        get_factors(0)


# ---------------------------------------------------------------------------
# get_greatest_common_divisor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "numbers, expected",
    [
        ((18, 45), 9),
        ((23, 37), 1),
        ((2520, 8350), 10),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1),
        ((12, 18, 24), 6),
        ((7, 7, 7), 7),
        ((100, 10, 1000), 10),
        ((1, 1), 1),
        ((2, 4, 8, 16), 2),
    ],
)
def test_get_greatest_common_divisor_happy_path(numbers, expected):
    """The gcd of n positive integers is computed correctly."""
    assert get_greatest_common_divisor(*numbers) == expected


def test_get_greatest_common_divisor_single_number():
    """A single argument returns that number itself."""
    assert get_greatest_common_divisor(42) == 42


def test_get_greatest_common_divisor_single_one():
    """A single argument of 1 returns 1."""
    assert get_greatest_common_divisor(1) == 1


def test_get_greatest_common_divisor_coprime_numbers():
    """Coprime numbers have a gcd of 1."""
    assert get_greatest_common_divisor(17, 19, 23) == 1


@pytest.mark.parametrize(
    "numbers",
    [
        (-10, 20),
        (0, 5),
        (1.5, 2),
        ("1", 2, 3),
        (10, None),
    ],
)
def test_get_greatest_common_divisor_invalid_input_raises_exception(numbers):
    """Invalid inputs raise a generic Exception with the documented message."""
    with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
        get_greatest_common_divisor(*numbers)


def test_get_greatest_common_divisor_no_arguments_raises():
    """Calling with no arguments raises (unpacking an empty map fails)."""
    with pytest.raises(Exception):
        get_greatest_common_divisor()


def test_get_greatest_common_divisor_propagates_type_error_as_exception():
    """A TypeError from get_factors is re-raised as a generic Exception."""
    with pytest.raises(Exception) as exc_info:
        get_greatest_common_divisor(-5, 10)
    assert not isinstance(exc_info.value, TypeError)
    assert "numbers must be integer and greater than zero" in str(exc_info.value)