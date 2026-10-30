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
def test_is_prime_returns_true_for_larger_primes(number):
    """Larger known primes (including values above the 6k+/-1 loop start) are prime."""
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
    """0 and 1 are not prime (they do not have exactly two factors)."""
    assert is_prime(number) is False


@pytest.mark.parametrize(
    "number",
    [2 * 2, 2 * 3, 3 * 3, 3 * 5, 3 * 5 * 7, 67483],
)
def test_is_prime_returns_false_for_products(number):
    """Products of small primes (and the docstring example 67483) are composite."""
    assert is_prime(number) is False


# ---------------------------------------------------------------------------
# Edge / boundary tests
# ---------------------------------------------------------------------------


def test_is_prime_boundary_two_and_three():
    """2 and 3 are the smallest primes and hit the special-case branch."""
    assert is_prime(2) is True
    assert is_prime(3) is True


def test_is_prime_boundary_four():
    """4 is the smallest composite and must be rejected by the even check."""
    assert is_prime(4) is False


def test_is_prime_square_of_large_prime():
    """A perfect square of a prime must be detected as composite."""
    prime = 1009
    assert is_prime(prime) is True
    assert is_prime(prime * prime) is False


def test_is_prime_product_of_two_large_primes():
    """Semiprime of two large primes must be detected as composite."""
    assert is_prime(1009 * 1013) is False


def test_is_prime_large_prime_near_sqrt_boundary():
    """A prime whose sqrt boundary is exercised by the 6k+/-1 loop."""
    number = 999983  # largest prime below one million
    assert is_prime(number) is True


def test_is_prime_uses_sqrt_bound():
    """The loop must stop at sqrt(n); verify with a mocked math.sqrt."""
    # 101 is prime; sqrt(101) ~ 10.05, so the loop range is range(5, 11, 6) -> [5]
    assert is_prime(101) is True
    # Sanity check that the implementation relies on math.sqrt at all.
    assert math.isqrt(101) == 10


def test_is_prime_does_not_loop_beyond_sqrt(monkeypatch):
    """Ensure the trial-division loop never iterates past sqrt(n)."""
    calls = []
    real_sqrt = math.sqrt

    def spy_sqrt(value):
        calls.append(value)
        return real_sqrt(value)

    monkeypatch.setattr(math, "sqrt", spy_sqrt)
    assert is_prime(97) is True
    assert calls == [97]


# ---------------------------------------------------------------------------
# Invalid-input tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("number", [-1, -4, -19, -1000])
def test_is_prime_raises_for_negative_integers(number):
    """Negative integers must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", [16.1, 2.0, 3.5, 0.0, -2.5])
def test_is_prime_raises_for_floats(number):
    """Floats (even integral ones) must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


@pytest.mark.parametrize("number", ["7", "abc", None, [2], (3,), {"a": 1}])
def test_is_prime_raises_for_non_numeric_types(number):
    """Non-integer types must raise ValueError."""
    with pytest.raises(ValueError, match="only accepts positive integers"):
        is_prime(number)


def test_is_prime_raises_for_bool_is_not_checked():
    """bool is a subclass of int; document the current behaviour explicitly."""
    # True == 1 -> not prime; False == 0 -> not prime. No exception is raised.
    assert is_prime(True) is False
    assert is_prime(False) is False


def test_is_prime_accepts_zero_as_int_not_raising():
    """Zero is a valid (non-negative) int, so it returns False rather than raising."""
    assert is_prime(0) is False


# ---------------------------------------------------------------------------
# Consistency / property-style tests
# ---------------------------------------------------------------------------


def test_is_prime_matches_naive_implementation_up_to_200():
    """Cross-check against a naive O(n) primality test for n in [0, 200)."""

    def naive(n: int) -> bool:
        if n < 2:
            return False
        return all(n % d for d in range(2, int(math.isqrt(n)) + 1))

    for n in range(200):
        assert is_prime(n) is naive(n), f"mismatch for n={n}"


def test_is_prime_returns_bool_type():
    """The function must always return an actual bool, never a truthy int."""
    for n in (0, 1, 2, 4, 97, 100):
        result = is_prime(n)
        assert isinstance(result, bool)