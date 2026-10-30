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
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 563, 2999, 7919],
)
def test_is_prime_returns_true_for_primes(number):
    """Known prime numbers must be reported as prime."""
    assert is_prime(number) is True


@pytest.mark.parametrize(
    "number",
    [0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 27, 49, 87, 100, 561, 67483],
)
def test_is_prime_returns_false_for_composites(number):
    """Known composite numbers (and 0/1) must be reported as non-prime."""
    assert is_prime(number) is False


# --------------------------------------------------------------------------- #
# Edge / boundary cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, False),  # smallest non-negative non-prime
        (1, False),  # neither prime nor composite
        (2, True),   # smallest prime
        (3, True),   # second prime, handled by the 1 < n < 4 shortcut
        (4, False),  # smallest composite
        (5, True),   # first number reaching the 6k +/- 1 loop
        (9, False),  # multiple of 3 caught before the loop
        (25, False), # 5 * 5, caught inside the loop
        (49, False), # 7 * 7, caught inside the loop
    ],
)
def test_is_prime_boundary_values(number, expected):
    """Boundary values around the special-cased branches behave correctly."""
    assert is_prime(number) is expected


def test_is_prime_large_prime():
    """A large prime exercises the full O(sqrt(n)) loop without early exit."""
    large_prime = 1_000_000_007
    assert is_prime(large_prime) is True


def test_is_prime_large_composite_with_large_factor():
    """A large composite whose smallest factor is large must be detected."""
    # 1000003 is prime, so its square is composite with a large smallest factor.
    assert is_prime(1_000_003 * 1_000_003) is False


def test_is_prime_square_of_prime():
    """Perfect squares of primes must be rejected by the loop."""
    assert is_prime(101 * 101) is False


def test_is_prime_product_of_two_large_primes():
    """Semiprimes of two large primes must be rejected."""
    assert is_prime(1009 * 1013) is False


# --------------------------------------------------------------------------- #
# Invalid input handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [-1, -4, -19, -1000])
def test_is_prime_raises_for_negative_numbers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -2.5])
def test_is_prime_raises_for_floats(number):
    """Floats (even integral ones) must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [2], (3,), {"n": 5}])
def test_is_prime_raises_for_non_integer_types(number):
    """Non-integer types must raise ValueError rather than TypeError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_accepts_bool_as_int_subclass():
    """bool is a subclass of int, so True (1) and False (0) are non-prime."""
    assert is_prime(True) is False
    assert is_prime(False) is False


# --------------------------------------------------------------------------- #
# Implementation-detail / isolation tests
# --------------------------------------------------------------------------- #
def test_is_prime_uses_sqrt_bound():
    """The loop bound must be derived from math.sqrt of the input."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(97) is True
    mock_sqrt.assert_called_once_with(97)


def test_is_prime_does_not_call_sqrt_for_small_numbers():
    """Numbers handled by the early branches must not compute a square root."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(2) is True
        assert is_prime(3) is True
        assert is_prime(4) is False
        assert is_prime(9) is False
    mock_sqrt.assert_not_called()


def test_is_prime_returns_bool_type():
    """The return value must be a real bool, not a truthy/falsy int."""
    assert isinstance(is_prime(13), bool)
    assert isinstance(is_prime(14), bool)


def test_is_prime_is_deterministic():
    """Repeated calls with the same input must yield the same result."""
    results = {is_prime(104_729) for _ in range(5)}
    assert results == {True}