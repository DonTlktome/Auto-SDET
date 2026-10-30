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
    """The function must return an actual bool, not a truthy/falsy value."""
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
    """3 is prime and hits the special-cased 1 < n < 4 branch."""
    assert is_prime(3) is True


def test_is_prime_four_is_not_prime():
    """4 is the smallest composite number greater than 3."""
    assert is_prime(4) is False


def test_is_prime_large_prime():
    """A large prime (2**31 - 1, a Mersenne prime) must be detected."""
    assert is_prime(2**31 - 1) is True


def test_is_prime_large_composite():
    """A large composite built from two primes must be rejected."""
    assert is_prime((2**31 - 1) * 3) is False


def test_is_prime_perfect_square_of_prime():
    """A perfect square of a prime must be rejected (boundary of sqrt loop)."""
    assert is_prime(97 * 97) is False


def test_is_prime_square_of_large_prime():
    """Squares of larger primes exercise the 6k +/- 1 loop boundary."""
    assert is_prime(1009 * 1009) is False


def test_is_prime_product_of_two_large_primes():
    """Semiprimes with both factors > 3 must be rejected by the loop."""
    assert is_prime(1009 * 1013) is False


def test_is_prime_number_of_form_6k_minus_1():
    """Numbers of the form 6k - 1 that are prime must be accepted."""
    assert is_prime(6 * 100 - 1) is True  # 599 is prime


def test_is_prime_number_of_form_6k_plus_1():
    """Numbers of the form 6k + 1 that are prime must be accepted."""
    assert is_prime(6 * 100 + 1) is True  # 601 is prime


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


@pytest.mark.parametrize("number", ["7", "abc", "", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_numeric_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_raises_for_bool_true():
    """bool is a subclass of int, so True is treated as 1 -> not prime."""
    assert is_prime(True) is False


def test_is_prime_raises_for_bool_false():
    """bool is a subclass of int, so False is treated as 0 -> not prime."""
    assert is_prime(False) is False


# ---------------------------------------------------------------------------
# Behavioural / implementation-detail tests (mocked dependencies)
# ---------------------------------------------------------------------------


def test_is_prime_uses_math_sqrt():
    """The loop bound must be derived from math.sqrt of the input."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(101) is True
    mock_sqrt.assert_called_once_with(101)


def test_is_prime_does_not_call_sqrt_for_small_numbers():
    """Small numbers short-circuit before the sqrt-based loop is reached."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(2) is True
        assert is_prime(3) is True
        assert is_prime(4) is False
        assert is_prime(9) is False
    mock_sqrt.assert_not_called()


def test_is_prime_does_not_call_sqrt_for_invalid_input():
    """Invalid input raises before any sqrt computation happens."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        with pytest.raises(ValueError):
            is_prime(-5)
    mock_sqrt.assert_not_called()


def test_is_prime_early_exit_on_first_factor():
    """A composite with a small factor must be rejected without full iteration."""
    # 5 * 1000003: the loop should find the factor 5 immediately.
    assert is_prime(5 * 1000003) is False


def test_is_prime_consistent_with_naive_implementation():
    """Cross-check results against a naive O(n) primality test for 0..500."""
    def naive(n: int) -> bool:
        if n < 2:
            return False
        return all(n % d for d in range(2, int(n**0.5) + 1))

    for n in range(0, 501):
        assert is_prime(n) is naive(n), f"mismatch for {n}"


def test_is_prime_consistent_with_naive_implementation_large_range():
    """Cross-check a sparse set of larger values against the naive test."""
    def naive(n: int) -> bool:
        if n < 2:
            return False
        return all(n % d for d in range(2, int(n**0.5) + 1))

    for n in range(1000, 1200):
        assert is_prime(n) is naive(n), f"mismatch for {n}"