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
        (1024, Counter({2: 10})),
    ],
)
def test_get_factors_happy_path(number, expected):
    """get_factors returns the correct prime factorisation for positive ints."""
    assert get_factors(number) == expected


def test_get_factors_one_returns_counter_with_one():
    """The special case number == 1 returns Counter({1: 1})."""
    assert get_factors(1) == Counter({1: 1})


def test_get_factors_prime_number_returns_itself():
    """A prime number's only factor is itself with exponent 1."""
    assert get_factors(97) == Counter({97: 1})


def test_get_factors_accepts_prefilled_counter():
    """A pre-supplied Counter is mutated/extended rather than discarded."""
    seed = Counter({2: 1})
    result = get_factors(4, seed)
    assert result[2] == 3  # 1 from seed + 2 from factorising 4


def test_get_factors_accepts_custom_starting_factor():
    """Starting the search at a higher factor skips smaller primes."""
    # 45 = 3 * 3 * 5; starting at 3 yields the same result as starting at 2
    assert get_factors(45, None, 3) == Counter({3: 2, 5: 1})


@pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, 2.0, "10", None])
def test_get_factors_invalid_input_raises_type_error(bad_value):
    """Non-positive or non-integer inputs raise TypeError."""
    with pytest.raises(TypeError, match="number must be integer and greater than zero"):
        get_factors(bad_value)


def test_get_factors_bool_true_is_treated_as_one():
    """bool is a subclass of int; True behaves like 1."""
    assert get_factors(True) == Counter({1: 1})


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
        ((36, 60, 84), 12),
    ],
)
def test_get_greatest_common_divisor_happy_path(numbers, expected):
    """The GCD of n positive integers is computed correctly."""
    assert get_greatest_common_divisor(*numbers) == expected


def test_get_greatest_common_divisor_single_number():
    """With a single argument the GCD is the number itself."""
    assert get_greatest_common_divisor(42) == 42


def test_get_greatest_common_divisor_single_one():
    """A single 1 yields 1."""
    assert get_greatest_common_divisor(1) == 1


def test_get_greatest_common_divisor_coprime_numbers():
    """Pairwise coprime numbers have a GCD of 1."""
    assert get_greatest_common_divisor(13, 17, 19, 23) == 1


def test_get_greatest_common_divisor_large_numbers():
    """Large inputs are handled without overflow or recursion issues."""
    assert get_greatest_common_divisor(1_000_000, 500_000) == 500_000


@pytest.mark.parametrize(
    "numbers",
    [
        (-10, 20),
        (0, 5),
        (1.5, 2),
        ("1", 2, 3),
        (None, 2),
    ],
)
def test_get_greatest_common_divisor_invalid_input_raises_exception(numbers):
    """Invalid inputs are wrapped into a generic Exception with a clear message."""
    with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
        get_greatest_common_divisor(*numbers)


def test_get_greatest_common_divisor_no_arguments_raises_value_error():
    """Calling with no arguments fails while unpacking the map result."""
    with pytest.raises(ValueError):
        get_greatest_common_divisor()


def test_get_greatest_common_divisor_propagates_type_error_as_exception():
    """A TypeError raised by get_factors is chained into the generic Exception."""
    with pytest.raises(Exception) as exc_info:
        get_greatest_common_divisor(-5, 10)
    assert isinstance(exc_info.value.__cause__, TypeError)