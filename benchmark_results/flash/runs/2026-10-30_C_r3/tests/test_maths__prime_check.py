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


@pytest.mark.parametrize("number", [2, 3])
def test_is_prime_smallest_primes(number):
    """2 and 3 are the smallest primes and hit the special-case branch."""
    assert is_prime(number) is True


def test_is_prime_four_is_not_prime():
    """4 is the smallest composite and is caught by the even-number branch."""
    assert is_prime(4) is False


def test_is_prime_nine_is_not_prime():
    """9 is the smallest odd composite divisible by 3."""
    assert is_prime(9) is False


def test_is_prime_large_prime():
    """A large prime exercises the 6k +/- 1 loop over many iterations."""
    assert is_prime(104729) is True  # 10000th prime


def test_is_prime_large_composite_with_large_factors():
    """A composite whose smallest factor is large must still be detected."""
    assert is_prime(104729 * 104729) is False


def test_is_prime_square_of_prime():
    """The square of a prime is composite and must be detected."""
    assert is_prime(97 * 97) is False


def test_is_prime_product_of_two_large_primes():
    """Semiprime of two large primes must be detected as composite."""
    assert is_prime(1009 * 1013) is False


def test_is_prime_loop_boundary_uses_sqrt():
    """The loop must stop at sqrt(n); verify with values around a square."""
    n = 49  # sqrt = 7
    assert is_prime(n) is False
    assert is_prime(n + 2) is False  # 51 == 3 * 17 is composite
    assert is_prime(n + 4) is True  # 53 is prime


# ---------------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("number", ["7", "abc", None, [7], (7,), {"n": 7}])
def test_is_prime_raises_for_non_numeric_types(number):
    """Non-integer types must raise ValueError rather than TypeError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_raises_for_bool_is_not_checked():
    """bool is a subclass of int; document the current behaviour explicitly."""
    # True == 1 -> not prime, False == 0 -> not prime
    assert is_prime(True) is False
    assert is_prime(False) is False


# ---------------------------------------------------------------------------
# Isolation / implementation-detail tests
# ---------------------------------------------------------------------------


def test_is_prime_does_not_call_math_sqrt_for_trivial_inputs():
    """Trivial inputs short-circuit before any sqrt computation."""
    with patch("prime_check.math.sqrt", side_effect=AssertionError("sqrt called")):
        assert is_prime(0) is False
        assert is_prime(1) is False
        assert is_prime(2) is True
        assert is_prime(3) is True
        assert is_prime(4) is False
        assert is_prime(9) is False


def test_is_prime_uses_math_sqrt_for_larger_inputs():
    """For larger inputs the implementation must consult math.sqrt."""
    real_sqrt = math.sqrt
    calls = []

    def spy(value):
        calls.append(value)
        return real_sqrt(value)

    with patch("prime_check.math.sqrt", side_effect=spy):
        assert is_prime(101) is True

    assert calls == [101]


def test_is_prime_early_exit_on_first_factor():
    """A composite with a small factor must return False without full scanning."""
    with patch("prime_check.math.sqrt", side_effect=AssertionError("sqrt called")):
        # 25 is odd and not divisible by 3, so it reaches the loop; but the
        # loop needs sqrt, so this asserts the even/3 branches only.
        assert is_prime(6) is False
        assert is_prime(12) is False
        assert is_prime(15) is False


# ---------------------------------------------------------------------------
# Consistency / property-style checks
# ---------------------------------------------------------------------------


def test_is_prime_matches_naive_implementation_for_small_range():
    """Cross-check against a naive trial-division implementation."""

    def naive(n: int) -> bool:
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True

    for n in range(0, 500):
        assert is_prime(n) is naive(n), f"mismatch at n={n}"


def test_is_prime_is_deterministic():
    """Repeated calls with the same input must yield the same result."""
    results = {is_prime(97) for _ in range(5)}
    assert results == {True}
