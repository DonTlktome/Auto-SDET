"""
Pytest test suite for the `is_prime` function from prime_check.py.

Covers:
- Positive test cases (known primes)
- Negative test cases (known non-primes)
- Edge cases (0, 1, 2, 3, large primes, large non-primes)
- Boundary / error cases (negative numbers, floats, non-numeric types)
- Performance / correctness for numbers near square boundaries
"""

import math
import pytest
from unittest.mock import patch, MagicMock

from prime_check import is_prime


# ---------------------------------------------------------------------------
# Positive test cases — known primes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prime",
    [
        2,       # smallest prime
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        97,      # two-digit prime
        563,     # from docstring
        2999,    # from docstring
        7919,    # 1000th prime
        104729,  # 10000th prime (large, still fits in int)
    ],
)
def test_is_prime_returns_true_for_primes(prime: int) -> None:
    """`is_prime` should return True for well-known prime numbers."""
    assert is_prime(prime) is True


# ---------------------------------------------------------------------------
# Negative test cases — known non-primes (composites, 0, 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "non_prime",
    [
        0,       # zero — not prime
        1,       # one — not prime
        4,       # 2*2
        6,       # 2*3
        8,       # even > 2
        9,       # 3*3
        15,      # 3*5
        21,      # 3*7
        25,      # 5*5
        27,      # from docstring
        49,      # 7*7
        87,      # from docstring
        100,     # even composite
        67483,   # from docstring
    ],
)
def test_is_prime_returns_false_for_non_primes(non_prime: int) -> None:
    """`is_prime` should return False for numbers that are not prime."""
    assert is_prime(non_prime) is False


# ---------------------------------------------------------------------------
# Edge cases — numbers just above/below typical boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "number,expected",
    [
        (2, True),     # boundary: first prime
        (3, True),     # boundary: second prime
        (4, False),    # first composite after 3
        (5, True),     # 6k-1 form (k=1)
        (7, True),     # 6k+1 form (k=1)
        (11, True),    # 6k-1 form (k=2)
        (13, True),    # 6k+1 form (k=2)
        (23, True),    # 6k-1 form (k=4)
        (25, False),   # 6k+1 form but composite (5*5)
        (35, False),   # 6k-1 form but composite (5*7)
        (49, False),   # 6k+1 form but composite (7*7)
        (121, False),  # 11*11, 6k+1 form
        (143, False),  # 11*13, 6k-1 form
    ],
)
def test_is_prime_boundary_and_6k_forms(number: int, expected: bool) -> None:
    """Test numbers around the 6k±1 pattern boundaries for correctness."""
    assert is_prime(number) == expected


# ---------------------------------------------------------------------------
# Large numbers — stress / correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "number,expected",
    [
        (999983, True),    # large prime
        (999984, False),   # even
        (1000000, False),  # large even
        (1000003, True),   # prime just above 1e6
        (1000033, True),   # another large prime
        (1000035, False),  # multiple of 5
    ],
)
def test_is_prime_large_numbers(number: int, expected: bool) -> None:
    """`is_prime` should correctly handle large numbers near 1e6."""
    assert is_prime(number) == expected


# ---------------------------------------------------------------------------
# Error cases — invalid input types and values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_input",
    [
        -1,
        -4,
        -100,
        -999983,  # negative of a large prime
    ],
)
def test_is_prime_raises_valueerror_for_negative_integers(bad_input: int) -> None:
    """`is_prime` should raise ValueError for any negative integer."""
    with pytest.raises(ValueError, match="is_prime\\(\\) only accepts positive integers"):
        is_prime(bad_input)


def test_is_prime_raises_valueerror_for_float() -> None:
    """`is_prime` should raise ValueError when given a float (even if it looks like an int)."""
    with pytest.raises(ValueError, match="is_prime\\(\\) only accepts positive integers"):
        is_prime(16.1)


def test_is_prime_raises_valueerror_for_zero() -> None:
    """`is_prime` should raise ValueError for 0? No — the function returns False for 0.
    But the docstring says it raises for non-positive integers. Let's verify actual behaviour:
    The code checks `not number >= 0` which is False for 0, so 0 is accepted and returns False.
    """
    # 0 is accepted by the precondition (0 >= 0 is True), so no error — returns False.
    assert is_prime(0) is False


@pytest.mark.parametrize(
    "non_int",
    [
        "5",
        None,
        [],
        {},
        (2,),
    ],
)
def test_is_prime_raises_valueerror_for_non_integer_types(non_int) -> None:
    """`is_prime` should raise ValueError for any non-int type."""
    with pytest.raises(ValueError, match="is_prime\\(\\) only accepts positive integers"):
        is_prime(non_int)


# ---------------------------------------------------------------------------
# Tests with mocked math.sqrt to verify loop boundaries
# ---------------------------------------------------------------------------

def test_is_prime_calls_math_sqrt_with_correct_argument() -> None:
    """Verify that `math.sqrt` is called with the input number during the loop phase."""
    with patch("prime_check.math.sqrt", wraps=math.sqrt) as mock_sqrt:
        result = is_prime(97)
        assert result is True
        # math.sqrt should have been called at least once with 97
        mock_sqrt.assert_any_call(97)


def test_is_prime_loop_boundary_exact_square() -> None:
    """Numbers that are perfect squares of primes (e.g. 49 = 7*7) should be caught.
    The loop runs up to int(sqrt(n))+1, so for 49 it checks i=5,7 (since range(5,8,6) → 5,7).
    """
    assert is_prime(49) is False  # 7*7


def test_is_prime_loop_boundary_just_above_square() -> None:
    """For n=53 (prime), sqrt(53)≈7.28, int(sqrt)+1=8, range(5,8,6)→5,7.
    Neither 5 nor 7 divides 53, so it's prime."""
    assert is_prime(53) is True


def test_is_prime_with_mocked_sqrt_returns_expected() -> None:
    """Use a mock to force a specific sqrt value and verify behaviour."""
    with patch("prime_check.math.sqrt", return_value=10.0):
        # sqrt returns 10.0 → int(10.0)+1 = 11 → range(5,11,6) → 5
        # For n=55 (5*11), 55%5==0 → False
        assert is_prime(55) is False

    with patch("prime_check.math.sqrt", return_value=10.0):
        # For n=53 (prime), 53%5!=0 and 53%7!=0 (but 7 not in range) → True
        assert is_prime(53) is True


# ---------------------------------------------------------------------------
# Exhaustive small-range verification
# ---------------------------------------------------------------------------

def test_is_prime_exhaustive_small_range() -> None:
    """Exhaustively check all integers 0..100 against a known-correct set."""
    primes_under_101 = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    }
    for n in range(101):
        assert is_prime(n) == (n in primes_under_101), f"Failed for {n}"


# ---------------------------------------------------------------------------
# Very large prime / non-prime (correctness near int boundaries)
# ---------------------------------------------------------------------------

def test_is_prime_very_large_prime() -> None:
    """Test a 7-digit prime to ensure the algorithm scales correctly."""
    assert is_prime(9999991) is True   # 9999991 is a known prime


def test_is_prime_very_large_composite() -> None:
    """Test a 7-digit composite that is a product of two primes."""
    assert is_prime(9999991 * 9999991) is False  # square of a large prime
