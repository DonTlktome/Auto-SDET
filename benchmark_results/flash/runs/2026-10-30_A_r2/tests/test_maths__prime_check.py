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
# Happy path tests
# --------------------------------------------------------------------------- #
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
    [4, 6, 8, 9, 10, 12, 15, 21, 25, 27, 49, 87, 121, 169, 1001],
)
def test_is_prime_returns_false_for_composites(number):
    """Composite numbers must be reported as non-prime."""
    assert is_prime(number) is False


@pytest.mark.parametrize(
    "number",
    [67483, 3 * 5 * 7, 2 * 2, 2 * 3, 3 * 3, 3 * 5],
)
def test_is_prime_returns_false_for_products(number):
    """Products of small factors must be reported as non-prime."""
    assert is_prime(number) is False


# --------------------------------------------------------------------------- #
# Edge / boundary cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [0, 1])
def test_is_prime_returns_false_for_zero_and_one(number):
    """0 and 1 have fewer than two factors, so they are not prime."""
    assert is_prime(number) is False


def test_is_prime_returns_true_for_two_and_three_boundary():
    """2 and 3 are the smallest primes and hit the special-case branch."""
    assert is_prime(2) is True
    assert is_prime(3) is True


def test_is_prime_returns_false_for_four_boundary():
    """4 is the first composite after the special-case branch."""
    assert is_prime(4) is False


def test_is_prime_returns_false_for_square_of_large_prime():
    """A perfect square of a prime must be detected as composite."""
    prime = 1009
    assert is_prime(prime) is True
    assert is_prime(prime * prime) is False


def test_is_prime_returns_false_for_product_of_two_large_primes():
    """Semiprimes of two large primes must be detected as composite."""
    assert is_prime(1009 * 1013) is False


def test_is_prime_returns_true_for_largest_prime_below_sqrt_boundary():
    """A prime just below a perfect-square boundary is still prime."""
    # 48**2 = 2304; 2303 = 7 * 7 * 47 -> composite, 2297 is prime.
    assert is_prime(2297) is True
    assert is_prime(2303) is False


# --------------------------------------------------------------------------- #
# Invalid input handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [-1, -4, -19, -1000])
def test_is_prime_raises_for_negative_integers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -2.0])
def test_is_prime_raises_for_floats(number):
    """Floats (even integral-valued ones) must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_integer_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_accepts_bool_as_int_subclass():
    """bool is a subclass of int; True == 1 (not prime), False == 0 (not prime)."""
    assert is_prime(True) is False
    assert is_prime(False) is False


def test_is_prime_accepts_large_int_type():
    """Arbitrarily large Python ints are accepted by the type check.

    Kept modest on purpose: the implementation is O(sqrt(n)) trial division,
    so a 2**61-sized input would take minutes. 2**31 - 1 is a Mersenne prime
    and finishes in a few milliseconds.
    """
    big_prime = 2**31 - 1  # 2147483647, Mersenne prime
    assert is_prime(big_prime) is True


# --------------------------------------------------------------------------- #
# Behavioural / implementation-detail tests (mocked)
# --------------------------------------------------------------------------- #
def test_is_prime_uses_sqrt_bound_for_loop():
    """The trial-division loop must be bounded by sqrt(n) (O(sqrt(n)) claim)."""
    number = 1_000_003  # prime
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(number) is True
    mock_sqrt.assert_called_once_with(number)


def test_is_prime_short_circuits_for_multiples_of_two():
    """Even numbers must return False without entering the trial-division loop."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(1_000_000) is False
    mock_sqrt.assert_not_called()


def test_is_prime_short_circuits_for_multiples_of_three():
    """Multiples of three must return False without entering the loop."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(999_999) is False
    mock_sqrt.assert_not_called()


def test_is_prime_short_circuits_for_small_values():
    """Values below 4 must be handled before any sqrt computation."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(0) is False
        assert is_prime(1) is False
        assert is_prime(2) is True
        assert is_prime(3) is True
    mock_sqrt.assert_not_called()


def test_is_prime_returns_bool_type():
    """The function must always return an actual bool, not a truthy value."""
    for value in (0, 1, 2, 4, 7, 9, 97, 100):
        result = is_prime(value)
        assert isinstance(result, bool)


# --------------------------------------------------------------------------- #
# Cross-check against a naive reference implementation
# --------------------------------------------------------------------------- #
def _naive_is_prime(n: int) -> bool:
    """Reference implementation: trial division over all integers up to n."""
    if n < 2:
        return False
    for divisor in range(2, int(n**0.5) + 1):
        if n % divisor == 0:
            return False
    return True


@pytest.mark.parametrize("number", list(range(0, 300)))
def test_is_prime_matches_naive_reference(number):
    """is_prime must agree with a naive reference for all n in [0, 300)."""
    assert is_prime(number) is _naive_is_prime(number)
