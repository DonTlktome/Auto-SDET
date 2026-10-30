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
    [4, 6, 8, 9, 10, 12, 15, 21, 25, 27, 49, 87, 91, 121, 169],
)
def test_is_prime_returns_false_for_composites(number):
    """Composite numbers must be reported as not prime."""
    assert is_prime(number) is False


@pytest.mark.parametrize("number", [0, 1])
def test_is_prime_returns_false_for_zero_and_one(number):
    """0 and 1 have fewer than two factors, so they are not prime."""
    assert is_prime(number) is False


@pytest.mark.parametrize("number", [67483, 1000003 * 2, 999999])
def test_is_prime_returns_false_for_large_composites(number):
    """Large composite numbers must be reported as not prime."""
    assert is_prime(number) is False


# --------------------------------------------------------------------------- #
# Edge cases / boundaries
# --------------------------------------------------------------------------- #
def test_is_prime_boundary_two_and_three():
    """The special-cased range 1 < n < 4 must yield True for 2 and 3 only."""
    assert is_prime(2) is True
    assert is_prime(3) is True
    assert is_prime(4) is False


def test_is_prime_square_of_prime_is_not_prime():
    """A prime squared is composite; exercises the 6k +/- 1 loop boundary."""
    for prime in (5, 7, 11, 13, 101, 1009):
        assert is_prime(prime * prime) is False


def test_is_prime_product_of_two_primes_is_not_prime():
    """Semiprimes must be detected as composite."""
    assert is_prime(5 * 7) is False
    assert is_prime(11 * 13) is False
    assert is_prime(3 * 5 * 7) is False


def test_is_prime_largest_prime_below_sqrt_boundary():
    """A prime just below a perfect square boundary must still be prime."""
    # 48**2 = 2304; 2303 = 7 * 329 is composite, 2297 is prime.
    assert is_prime(2297) is True
    assert is_prime(2303) is False


def test_is_prime_accepts_bool_as_int_subclass():
    """bool is a subclass of int, so True (==1) is not prime and False (==0) is not prime."""
    assert is_prime(True) is False
    assert is_prime(False) is False


def test_is_prime_handles_very_large_prime():
    """A large prime exercises the full O(sqrt(n)) loop."""
    assert is_prime(2**31 - 1) is True  # Mersenne prime 2147483647


# --------------------------------------------------------------------------- #
# Invalid input handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [-1, -4, -19, -1000])
def test_is_prime_raises_for_negative_integers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -1.5])
def test_is_prime_raises_for_floats(number):
    """Floats (even integral ones) must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_numeric_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_raises_for_complex_number():
    """Complex numbers are not ints and must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(3 + 4j)


# --------------------------------------------------------------------------- #
# Isolation / implementation-detail tests using mocks
# --------------------------------------------------------------------------- #
def test_is_prime_uses_math_sqrt():
    """The implementation should rely on math.sqrt for its loop bound."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(97) is True
    mock_sqrt.assert_called_once_with(97)


def test_is_prime_does_not_call_sqrt_for_trivial_inputs():
    """Trivial inputs (0, 1, 2, 3, even, multiples of 3) short-circuit before sqrt."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        for number in (0, 1, 2, 3, 4, 9, 27):
            is_prime(number)
    mock_sqrt.assert_not_called()


def test_is_prime_short_circuits_on_first_divisor():
    """The loop must stop at the first divisor found (no full scan)."""
    # 25 = 5 * 5 -> divisor found at i = 5, the very first loop iteration.
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(25) is False
    mock_sqrt.assert_called_once_with(25)


def test_is_prime_returns_bool_type():
    """The return value must be a real bool, not a truthy int."""
    assert isinstance(is_prime(13), bool)
    assert isinstance(is_prime(14), bool)