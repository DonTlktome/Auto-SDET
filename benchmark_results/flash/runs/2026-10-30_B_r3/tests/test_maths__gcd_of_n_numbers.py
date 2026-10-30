"""
Tests for maths/gcd_of_n_numbers.py

Covers:
    - get_factors
    - get_greatest_common_divisor
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maths"))

from gcd_of_n_numbers import (  # noqa: E402
    get_factors,
    get_greatest_common_divisor,
)


# ---------------------------------------------------------------------------
# get_factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (2, Counter({2: 1})),
        (3, Counter({3: 1})),
        (4, Counter({2: 2})),
        (12, Counter({2: 2, 3: 1})),
        (45, Counter({3: 2, 5: 1})),
        (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
        (23, Counter({23: 1})),
        (97, Counter({97: 1})),
        (1024, Counter({2: 10})),
    ],
)
def test_get_factors_happy_path(number, expected):
    """get_factors returns the correct prime-factorization Counter."""
    assert get_factors(number) == expected


def test_get_factors_one_returns_counter_with_one():
    """The special case number == 1 returns Counter({1: 1})."""
    assert get_factors(1) == Counter({1: 1})


def test_get_factors_reconstructs_original_number():
    """Multiplying factor**power must reproduce the original number."""
    number = 360
    factors = get_factors(number)
    product = 1
    for base, power in factors.items():
        product *= base**power
    assert product == number


def test_get_factors_accepts_prepopulated_counter():
    """A caller-supplied Counter is used and mutated in place."""
    seed = Counter({2: 1})
    result = get_factors(12, factors=seed)
    assert result == Counter({2: 3, 3: 1})
    # The same object should have been reused.
    assert result is seed


def test_get_factors_accepts_custom_starting_factor():
    """Starting the search at a higher factor skips smaller primes."""
    # 45 = 3 * 3 * 5; starting at 3 yields the same factorization.
    assert get_factors(45, factor=3) == Counter({3: 2, 5: 1})


@pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, 2.0, "12", None])
def test_get_factors_rejects_invalid_input(bad_value):
    """Non-positive or non-integer inputs raise TypeError."""
    with pytest.raises(TypeError, match="number must be integer and greater than zero"):
        get_factors(bad_value)


def test_get_factors_bool_is_treated_as_int():
    """Booleans are ints in Python; True behaves like 1."""
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
        ((100, 100, 100), 100),
        ((7,), 7),
        ((1,), 1),
        ((2, 4, 8, 16), 2),
        ((36, 60, 84), 12),
    ],
)
def test_get_greatest_common_divisor_happy_path(numbers, expected):
    """The GCD of the given numbers matches the expected value."""
    assert get_greatest_common_divisor(*numbers) == expected


def test_get_greatest_common_divisor_single_number_returns_itself():
    """A single argument yields that number (boundary case)."""
    assert get_greatest_common_divisor(42) == 42


def test_get_greatest_common_divisor_coprime_numbers():
    """Coprime numbers have a GCD of exactly 1."""
    assert get_greatest_common_divisor(17, 19, 23) == 1


def test_get_greatest_common_divisor_result_divides_all_inputs():
    """The returned GCD must divide every input without remainder."""
    numbers = (48, 72, 120)
    gcd = get_greatest_common_divisor(*numbers)
    assert all(n % gcd == 0 for n in numbers)


@pytest.mark.parametrize(
    "numbers",
    [
        (-10, 20),
        (0, 5),
        (1.5, 2),
        ("1", 2, 3),
        (5, None),
    ],
)
def test_get_greatest_common_divisor_rejects_invalid_input(numbers):
    """Invalid inputs raise a generic Exception with the documented message."""
    with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
        get_greatest_common_divisor(*numbers)


def test_get_greatest_common_divisor_no_arguments_raises():
    """Calling with no arguments raises (unpacking an empty map fails)."""
    with pytest.raises(Exception):
        get_greatest_common_divisor()


def test_get_greatest_common_divisor_propagates_type_error_as_exception():
    """A TypeError from get_factors is re-raised as a plain Exception."""
    with pytest.raises(Exception) as exc_info:
        get_greatest_common_divisor(-5, 10)
    assert not isinstance(exc_info.value, TypeError)
    assert "numbers must be integer and greater than zero" in str(exc_info.value)