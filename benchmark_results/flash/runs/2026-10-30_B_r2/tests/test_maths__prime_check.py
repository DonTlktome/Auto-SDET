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
    [0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 27, 87, 100, 105, 67483],
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


def test_is_prime_two_and_three_are_prime():
    """The smallest primes (2 and 3) hit the special-case branch."""
    assert is_prime(2) is True
    assert is_prime(3) is True


def test_is_prime_four_is_not_prime():
    """4 is the first composite above the special-case branch."""
    assert is_prime(4) is False


def test_is_prime_large_prime():
    """A large prime exercises the 6k +/- 1 loop over many iterations."""
    assert is_prime(104729) is True  # 10000th prime


def test_is_prime_large_composite_with_large_factor():
    """A composite whose smallest factor is large must still be detected."""
    assert is_prime(104729 * 104729) is False


def test_is_prime_square_of_prime():
    """Perfect squares of primes must be detected as composite."""
    assert is_prime(7 * 7) is False


def test_is_prime_product_of_two_primes():
    """Semiprimes must be detected as composite."""
    assert is_prime(3 * 5 * 7) is False


def test_is_prime_returns_bool_type():
    """The function must return an actual bool, not a truthy value."""
    assert isinstance(is_prime(13), bool)
    assert isinstance(is_prime(14), bool)


# --------------------------------------------------------------------------- #
# Invalid input handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [-1, -4, -19, -2, -100])
def test_is_prime_raises_for_negative_numbers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -1.0])
def test_is_prime_raises_for_floats(number):
    """Floats (even integral ones) must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_integer_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_accepts_bool_as_int_subclass():
    """bool is a subclass of int, so True (==1) is not prime and False (==0) is not prime."""
    assert is_prime(True) is False
    assert is_prime(False) is False


# --------------------------------------------------------------------------- #
# Isolation / implementation-detail tests
# --------------------------------------------------------------------------- #
def test_is_prime_uses_sqrt_bound():
    """The loop bound must be derived from math.sqrt of the input."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(97) is True
    mock_sqrt.assert_called_once_with(97)


def test_is_prime_does_not_call_sqrt_for_small_numbers():
    """Numbers handled by the early-return branches must not compute a sqrt."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(2) is True
        assert is_prime(9) is False
        assert is_prime(1) is False
    mock_sqrt.assert_not_called()


def test_is_prime_does_not_call_sqrt_for_invalid_input():
    """Invalid input must raise before any sqrt computation happens."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        with pytest.raises(ValueError):
            is_prime(-5)
    mock_sqrt.assert_not_called()


def test_is_prime_early_exit_on_small_factor():
    """A composite with a small factor must be rejected without scanning far."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        assert is_prime(5 * 1000003) is False
    mock_sqrt.assert_called_once()