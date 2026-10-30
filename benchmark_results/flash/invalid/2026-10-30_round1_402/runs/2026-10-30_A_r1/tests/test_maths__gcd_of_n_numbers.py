"""
Tests for maths/gcd_of_n_numbers.py

Covers:
    - get_factors
    - get_greatest_common_divisor
"""

from collections import Counter

import pytest

from maths.gcd_of_n_numbers import get_factors, get_greatest_common_divisor


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
        (97, Counter({97: 1})),  # prime
    ],
)
def test_get_factors_happy_path(number, expected):
    """get_factors returns the correct prime-factor Counter for positive ints."""
    assert get_factors(number) == expected


def test_get_factors_one_returns_counter_with_one():
    """The special case number == 1 returns Counter({1: 1})."""
    assert get_factors(1) == Counter({1: 1})


def test_get_factors_accepts_prepopulated_counter():
    """A pre-supplied Counter is mutated/extended rather than discarded."""
    seed = Counter({2: 1})
    result = get_factors(4, factors=seed)
    # 4 = 2 * 2, plus the pre-existing 2 -> {2: 3}
    assert result == Counter({2: 3})


def test_get_factors_custom_starting_factor():
    """Starting from a factor that divides the number yields that factor."""
    # 9 with factor=3 -> 3 * 3
    assert get_factors(9, factor=3) == Counter({3: 2})


@pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, "10", None])
def test_get_factors_invalid_input_raises_type_error(bad_value):
    """Non-positive or non-integer inputs raise TypeError."""
    with pytest.raises(TypeError):
        get_factors(bad_value)


def test_get_factors_error_message():
    """The TypeError carries the documented message."""
    with pytest.raises(TypeError) as exc_info:
        get_factors(0)
    assert "number must be integer and greater than zero" in str(exc_info.value)


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
        ((100, 10), 10),
        ((1, 1), 1),
        ((2, 4, 8, 16), 2),
        ((36, 60, 84), 12),
    ],
)
def test_get_greatest_common_divisor_happy_path(numbers, expected):
    """GCD of n positive integers is computed correctly."""
    assert get_greatest_common_divisor(*numbers) == expected


def test_get_greatest_common_divisor_single_number():
    """A single argument returns that number itself."""
    assert get_greatest_common_divisor(42) == 42


def test_get_greatest_common_divisor_single_one():
    """A single argument of 1 returns 1."""
    assert get_greatest_common_divisor(1) == 1


def test_get_greatest_common_divisor_coprime_numbers():
    """Coprime numbers have a GCD of 1."""
    assert get_greatest_common_divisor(9, 28, 55) == 1


@pytest.mark.parametrize(
    "numbers",
    [
        (-10, 20),
        (0, 5),
        (1.5, 2),
        ("1", 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (10, None),
    ],
)
def test_get_greatest_common_divisor_invalid_input_raises_exception(numbers):
    """Invalid inputs raise a generic Exception with the documented message."""
    with pytest.raises(Exception) as exc_info:
        get_greatest_common_divisor(*numbers)
    assert "numbers must be integer and greater than zero" in str(exc_info.value)


def test_get_greatest_common_divisor_no_arguments_raises():
    """Calling with no arguments raises (unpacking of empty map fails)."""
    with pytest.raises(Exception):
        get_greatest_common_divisor()