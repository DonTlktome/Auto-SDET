"""Pytest test suite for maths/prime_check.py."""

import math
import sys
from pathlib import Path

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prime_check import is_prime  # noqa: E402


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number",
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
)
def test_is_prime_returns_true_for_small_primes(number):
    """Small known primes must be reported as prime."""
    assert is_prime(number) is True


@pytest.mark.parametrize(
    "number",
    [563, 2999, 7919, 104729],
)
def test_is_prime_returns_true_for_large_primes(number):
    """Larger known primes must be reported as prime."""
    assert is_prime(number) is True


@pytest.mark.parametrize(
    "number",
    [4, 6, 8, 9, 10, 12, 15, 21, 25, 27, 49, 87, 121, 169],
)
def test_is_prime_returns_false_for_composites(number):
    """Composite numbers must be reported as non-prime."""
    assert is_prime(number) is False


@pytest.mark.parametrize(
    "number",
    [67483, 3 * 5 * 7, 2 * 2, 2 * 3, 3 * 3, 1001, 9999],
)
def test_is_prime_returns_false_for_products(number):
    """Products of small factors must be reported as non-prime."""
    assert is_prime(number) is False


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [0, 1])
def test_is_prime_returns_false_for_zero_and_one(number):
    """0 and 1 have fewer than two factors, so they are not prime."""
    assert is_prime(number) is False


def test_is_prime_returns_false_for_two_squared_boundary():
    """The smallest composite (4) must be rejected."""
    assert is_prime(4) is False


def test_is_prime_returns_true_for_smallest_prime():
    """2 is the smallest prime and must be accepted."""
    assert is_prime(2) is True


def test_is_prime_handles_large_prime_near_square_boundary():
    """A prime whose square root is just below an integer boundary."""
    # 97 is prime; sqrt(97) ~ 9.85, so the loop must not falsely reject it.
    assert is_prime(97) is True


def test_is_prime_handles_square_of_large_prime():
    """The square of a prime must be detected as composite."""
    prime = 101
    assert is_prime(prime * prime) is False


def test_is_prime_handles_semiprime_of_two_large_primes():
    """A product of two large primes must be detected as composite."""
    assert is_prime(101 * 103) is False


def test_is_prime_handles_large_even_number():
    """A large even number must be rejected by the fast path."""
    assert is_prime(10**12) is False


def test_is_prime_handles_large_multiple_of_three():
    """A large multiple of three must be rejected by the fast path."""
    assert is_prime(3 * 10**11) is False


def test_is_prime_loop_boundary_uses_sqrt():
    """Verify the loop bound is derived from sqrt(n) (boundary behaviour)."""
    # 25 = 5 * 5; sqrt(25) == 5, so the loop must include i == 5.
    assert is_prime(25) is False
    # 49 = 7 * 7; sqrt(49) == 7.
    assert is_prime(49) is False


# ---------------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [-1, -2, -4, -19, -100])
def test_is_prime_raises_for_negative_integers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -1.5])
def test_is_prime_raises_for_floats(number):
    """Float inputs must raise ValueError, even if integral in value."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", "", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_integer_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_raises_for_bool_is_not_checked():
    """bool is a subclass of int; document the current behaviour explicitly."""
    # True == 1 -> not prime; False == 0 -> not prime. No exception is raised.
    assert is_prime(True) is False
    assert is_prime(False) is False


# ---------------------------------------------------------------------------
# Consistency / property-style checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", list(range(0, 200)))
def test_is_prime_matches_naive_implementation(number):
    """is_prime must agree with a naive trial-division reference for 0..199."""
    if number < 2:
        expected = False
    else:
        expected = all(number % d != 0 for d in range(2, int(math.isqrt(number)) + 1))
    assert is_prime(number) is expected


def test_is_prime_returns_actual_bool():
    """The function must return a real bool, not a truthy/falsy value."""
    assert isinstance(is_prime(7), bool)
    assert isinstance(is_prime(8), bool)