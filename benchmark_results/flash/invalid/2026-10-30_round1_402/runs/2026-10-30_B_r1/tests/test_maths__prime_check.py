"""Pytest test suite for maths/prime_check.py."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prime_check import is_prime  # noqa: E402


# --------------------------------------------------------------------------- #
# Positive / happy-path tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "number",
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 563, 2999],
)
def test_is_prime_returns_true_for_primes(number):
    """Known prime numbers must be reported as prime."""
    assert is_prime(number) is True


@pytest.mark.parametrize(
    "number",
    [0, 1, 4, 6, 9, 15, 16, 27, 87, 100, 105, 67483],
)
def test_is_prime_returns_false_for_composites(number):
    """Known composite numbers (and 0/1) must be reported as not prime."""
    assert is_prime(number) is False


# --------------------------------------------------------------------------- #
# Edge cases / boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [0, 1])
def test_is_prime_zero_and_one_are_not_prime(number):
    """0 and 1 have fewer than two factors, so they are not prime."""
    assert is_prime(number) is False


@pytest.mark.parametrize("number", [2, 3])
def test_is_prime_smallest_primes(number):
    """2 and 3 are the smallest primes and hit the early-return branch."""
    assert is_prime(number) is True


def test_is_prime_four_is_first_composite():
    """4 is the smallest composite number and must be rejected."""
    assert is_prime(4) is False


def test_is_prime_large_prime():
    """A large prime exercises the 6k +/- 1 loop over many iterations."""
    assert is_prime(104729) is True  # 10000th prime


def test_is_prime_large_composite_with_large_factors():
    """A composite whose smallest factor is large must still be detected."""
    assert is_prime(104729 * 104729) is False


def test_is_prime_square_of_prime():
    """The square of a prime is composite and must be detected by the loop."""
    assert is_prime(97 * 97) is False


def test_is_prime_product_of_two_large_primes():
    """Semiprime of two large primes must be reported as not prime."""
    assert is_prime(1009 * 1013) is False


def test_is_prime_returns_bool_type():
    """The function must return an actual bool, not a truthy value."""
    assert isinstance(is_prime(7), bool)
    assert isinstance(is_prime(8), bool)


# --------------------------------------------------------------------------- #
# Invalid input handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [-1, -4, -19, -2, -100])
def test_is_prime_raises_for_negative_numbers(number):
    """Negative integers are rejected with a ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -1.5])
def test_is_prime_raises_for_floats(number):
    """Floats are rejected even when they hold integral values."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_numeric_types(number):
    """Non-integer types are rejected with a ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_accepts_bool_as_int_subclass():
    """bool is a subclass of int, so True (1) and False (0) are handled as ints."""
    assert is_prime(True) is False  # True == 1, not prime
    assert is_prime(False) is False  # False == 0, not prime


def test_is_prime_error_message_is_exact():
    """The raised ValueError carries the documented message."""
    with pytest.raises(ValueError) as exc_info:
        is_prime(-4)
    assert str(exc_info.value) == "is_prime() only accepts positive integers"


# --------------------------------------------------------------------------- #
# Isolation / implementation-detail tests
# --------------------------------------------------------------------------- #
def test_is_prime_uses_math_sqrt():
    """The loop bound is derived from math.sqrt, which is called for big inputs."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(97) is True
    mock_sqrt.assert_called_once_with(97)


def test_is_prime_does_not_call_sqrt_for_small_numbers():
    """Numbers below 4 short-circuit before any sqrt computation."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(2) is True
        assert is_prime(3) is True
        assert is_prime(1) is False
    mock_sqrt.assert_not_called()


def test_is_prime_does_not_call_sqrt_for_even_numbers():
    """Even numbers are rejected before the sqrt-based loop is reached."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(100) is False
    mock_sqrt.assert_not_called()


def test_is_prime_does_not_call_sqrt_for_multiples_of_three():
    """Multiples of three are rejected before the sqrt-based loop is reached."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(87) is False
    mock_sqrt.assert_not_called()


# --------------------------------------------------------------------------- #
# Cross-check against a reference implementation
# --------------------------------------------------------------------------- #
def _reference_is_prime(n: int) -> bool:
    """Naive trial-division reference implementation."""
    if n < 2:
        return False
    for divisor in range(2, int(math.isqrt(n)) + 1):
        if n % divisor == 0:
            return False
    return True


@pytest.mark.parametrize("number", list(range(0, 300)))
def test_is_prime_matches_reference_implementation(number):
    """is_prime agrees with a naive reference implementation for 0..299."""
    assert is_prime(number) is _reference_is_prime(number)