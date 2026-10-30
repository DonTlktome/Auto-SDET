"""Tests for the prime_check module."""

import math
from unittest.mock import patch

import pytest

from maths.prime_check import is_prime


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number",
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 563, 2999],
)
def test_is_prime_returns_true_for_primes(number):
    """Known prime numbers must be reported as prime."""
    assert is_prime(number) is True


@pytest.mark.parametrize(
    "number",
    [0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 27, 87, 100, 105, 67483],
)
def test_is_prime_returns_false_for_composites(number):
    """Known composite numbers (and 0/1) must be reported as not prime."""
    assert is_prime(number) is False


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [0, 1])
def test_is_prime_zero_and_one_are_not_prime(number):
    """0 and 1 have fewer than two factors, so they are not prime."""
    assert is_prime(number) is False


@pytest.mark.parametrize("number", [2, 3])
def test_is_prime_smallest_primes(number):
    """2 and 3 are the smallest primes and hit the early-return branch."""
    assert is_prime(number) is True


def test_is_prime_four_is_not_prime():
    """4 is the smallest composite number and must not be prime."""
    assert is_prime(4) is False


def test_is_prime_large_prime():
    """A large prime exercises the 6k +/- 1 loop over many iterations."""
    assert is_prime(104729) is True  # 10000th prime


def test_is_prime_large_composite_with_large_factor():
    """A large composite whose smallest factor is > 3 must be detected."""
    assert is_prime(104729 * 104729) is False


def test_is_prime_perfect_square_of_prime():
    """The square of a prime must be detected as composite."""
    assert is_prime(97 * 97) is False


def test_is_prime_product_of_two_large_primes():
    """Semiprime of two large primes must be detected as composite."""
    assert is_prime(1009 * 1013) is False


def test_is_prime_returns_bool_type():
    """The function must return an actual bool, not a truthy value."""
    assert isinstance(is_prime(7), bool)
    assert isinstance(is_prime(8), bool)


# ---------------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [-1, -4, -19, -100])
def test_is_prime_raises_for_negative_numbers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -1.5])
def test_is_prime_raises_for_floats(number):
    """Floats (even integral ones) must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [2], (3,), {"a": 1}])
def test_is_prime_raises_for_non_int_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_accepts_bool_as_int_subclass():
    """bool is a subclass of int; True == 1 (not prime), False == 0 (not prime)."""
    assert is_prime(True) is False
    assert is_prime(False) is False


# ---------------------------------------------------------------------------
# Isolation / implementation-detail tests
# ---------------------------------------------------------------------------


def test_is_prime_uses_math_sqrt():
    """The loop bound is derived from math.sqrt; verify it is actually called."""
    with patch("math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(97) is True
    mock_sqrt.assert_called_once_with(97)


def test_is_prime_does_not_call_sqrt_for_small_numbers():
    """Numbers handled by the early-return branches must not call math.sqrt."""
    with patch("math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(2) is True
        assert is_prime(3) is True
        assert is_prime(4) is False
        assert is_prime(9) is False
    mock_sqrt.assert_not_called()


def test_is_prime_does_not_call_sqrt_for_invalid_input():
    """Invalid input must raise before any sqrt computation happens."""
    with patch("math.sqrt", wraps=math.sqrt) as mock_sqrt:
        with pytest.raises(ValueError):
            is_prime(-5)
    mock_sqrt.assert_not_called()


# ---------------------------------------------------------------------------
# Cross-check against a naive reference implementation
# ---------------------------------------------------------------------------


def _naive_is_prime(n: int) -> bool:
    """Simple reference implementation used to cross-check results."""
    if n < 2:
        return False
    for divisor in range(2, int(n**0.5) + 1):
        if n % divisor == 0:
            return False
    return True


@pytest.mark.parametrize("number", list(range(0, 200)))
def test_is_prime_matches_naive_implementation(number):
    """For 0..199 the result must match a naive trial-division reference."""
    assert is_prime(number) is _naive_is_prime(number)
