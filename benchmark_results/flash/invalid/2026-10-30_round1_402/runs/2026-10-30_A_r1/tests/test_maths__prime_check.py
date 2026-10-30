"""Pytest test suite for maths/prime_check.py."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prime_check import is_prime  # noqa: E402


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


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
    """Known composite numbers (and 0/1) must be reported as not prime."""
    assert is_prime(number) is False


def test_is_prime_returns_bool_type():
    """The return value must be a real bool, not a truthy/falsy int."""
    assert isinstance(is_prime(7), bool)
    assert isinstance(is_prime(8), bool)


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [0, 1])
def test_is_prime_zero_and_one_are_not_prime(number):
    """0 and 1 have fewer than two factors, so they are not prime."""
    assert is_prime(number) is False


def test_is_prime_two_is_smallest_prime():
    """2 is the smallest (and only even) prime."""
    assert is_prime(2) is True


def test_is_prime_three_is_prime():
    """3 is prime and exercises the `1 < number < 4` fast path."""
    assert is_prime(3) is True


def test_is_prime_four_is_not_prime():
    """4 is the smallest composite number greater than 3."""
    assert is_prime(4) is False


def test_is_prime_square_of_large_prime():
    """A perfect square of a prime must be detected as composite."""
    prime = 1009
    assert is_prime(prime) is True
    assert is_prime(prime * prime) is False


def test_is_prime_product_of_two_large_primes():
    """Semiprimes built from large primes must be detected as composite."""
    assert is_prime(1009 * 1013) is False


def test_is_prime_large_prime():
    """A large prime near the top of the loop range must be detected."""
    assert is_prime(104729) is True  # 10000th prime


def test_is_prime_large_composite_with_large_factor():
    """A composite whose smallest factor is large must still be detected."""
    assert is_prime(104729 * 3) is False


def test_is_prime_loop_boundary_multiple_of_six_minus_one():
    """Numbers of the form 6k-1 that are composite must be rejected."""
    # 35 = 5 * 7, 77 = 7 * 11, 143 = 11 * 13
    for number in (35, 77, 143, 323, 899):
        assert is_prime(number) is False


def test_is_prime_loop_boundary_multiple_of_six_plus_one():
    """Numbers of the form 6k+1 that are composite must be rejected."""
    # 25 = 5 * 5, 49 = 7 * 7, 121 = 11 * 11, 169 = 13 * 13
    for number in (25, 49, 121, 169, 289):
        assert is_prime(number) is False


# ---------------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [-1, -2, -4, -19, -1000])
def test_is_prime_raises_for_negative_integers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -4.0])
def test_is_prime_raises_for_floats(number):
    """Floats (even integral ones) must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_numeric_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_accepts_bool_as_int_subclass():
    """bool is a subclass of int, so True (==1) and False (==0) are accepted."""
    assert is_prime(True) is False  # True == 1, not prime
    assert is_prime(False) is False  # False == 0, not prime


def test_is_prime_raises_for_complex_number():
    """Complex numbers are not ints and must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(3 + 0j)


# ---------------------------------------------------------------------------
# Algorithmic behaviour / isolation
# ---------------------------------------------------------------------------


def test_is_prime_uses_sqrt_bound():
    """The trial-division loop must stop at sqrt(n), not iterate to n."""
    calls = []
    real_sqrt = math.sqrt

    def spy_sqrt(value):
        calls.append(value)
        return real_sqrt(value)

    with patch("prime_check.math.sqrt", side_effect=spy_sqrt):
        assert is_prime(104729) is True

    assert calls == [104729]


def test_is_prime_does_not_call_sqrt_for_small_numbers():
    """The sqrt fast paths for n < 4 and even/multiple-of-3 numbers skip sqrt."""
    with patch("prime_check.math.sqrt") as mock_sqrt:
        assert is_prime(2) is True
        assert is_prime(3) is True
        assert is_prime(0) is False
        assert is_prime(1) is False
        assert is_prime(4) is False
        assert is_prime(9) is False
    mock_sqrt.assert_not_called()


def test_is_prime_short_circuits_on_first_factor():
    """A composite must be rejected without scanning every candidate."""
    # 5 * 1000003 -> first candidate 5 divides it immediately.
    number = 5 * 1000003
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(number) is False
    mock_sqrt.assert_called_once_with(number)


def test_is_prime_matches_naive_reference_implementation():
    """Cross-check the optimised implementation against a naive sieve."""
    limit = 500
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for multiple in range(i * i, limit + 1, i):
                sieve[multiple] = False

    for number in range(limit + 1):
        assert is_prime(number) is sieve[number], f"mismatch at {number}"


def test_is_prime_is_deterministic():
    """Repeated calls with the same input must return the same result."""
    results = {is_prime(2999) for _ in range(5)}
    assert results == {True}